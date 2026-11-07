"""
A delta oracle for PepGate.

Instead of predicting absolute log Papp and subtracting, this predicts the
CHANGE in permeability caused by one backbone edit, from features of the
backbone context around the edit site. Three reasons it works better:

  - it never has to be calibrated, so the systematic offset that sinks the
    absolute predictor is irrelevant;
  - its features transfer across scaffolds, because none of them reference
    an absolute position or a specific side chain;
  - it is evaluated on the decision the search actually makes, which is
    "which of these edits is best", not "does this edit help".

Usage:
    from delta_oracle import DeltaOracle, build_pairs

    pairs = build_pairs(project_root)
    oracle = DeltaOracle().fit(pairs)
    oracle.score_edits(pattern, editable_positions)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

METHYL = ("NMe", "D+NMe")
INVERTED = ("D", "D+NMe")

STATES = ("none", "D", "NMe", "D+NMe")

FEATURE_NAMES = (
    "ring_size",
    "n_methyl",
    "n_D",
    "frac_methyl",
    "frac_D",
    "is_NMe_edit",
    "is_D_edit",
    "left_methyl",
    "right_methyl",
    "left_D",
    "right_D",
    "site_none",
    "site_D",
    "site_NMe",
    "dist_to_end",
)


def parse_pattern(value) -> list[str] | None:
    """Turn a stored backbone pattern into a list of state strings."""

    if isinstance(value, (list, tuple)):
        return [str(s).strip() for s in value]

    text = str(value).strip()

    if not text or text == "nan":
        return None

    return [s.strip() for s in text.strip("()").replace("'", "").split(",")]


def edit_features(pattern: list[str], position: int, edit_type: str):
    """
    Features for applying `edit_type` at `position` of `pattern`.

    Everything here is relative: counts, fractions, neighbour states and a
    symmetric distance-to-end. Nothing references an absolute index or a
    side-chain identity, so a model fitted on one scaffold can score another.
    """

    n = len(pattern)

    if not 0 <= position < n:
        return None

    n_methyl = sum(s in METHYL for s in pattern)
    n_inverted = sum(s in INVERTED for s in pattern)

    left = pattern[(position - 1) % n]
    right = pattern[(position + 1) % n]
    site = pattern[position]

    return [
        float(n),
        float(n_methyl),
        float(n_inverted),
        n_methyl / n,
        n_inverted / n,
        float(edit_type == "NMe"),
        float(edit_type == "D"),
        float(left in METHYL),
        float(right in METHYL),
        float(left in INVERTED),
        float(right in INVERTED),
        float(site == "none"),
        float(site == "D"),
        float(site == "NMe"),
        min(position, n - position) / n,
    ]


def apply_edit(pattern: list[str], position: int, edit_type: str) -> list[str]:
    """Return a new pattern with one backbone state toggled."""

    out = list(pattern)
    current = out[position]

    if edit_type == "NMe":
        out[position] = "D+NMe" if current in INVERTED else "NMe"
    elif edit_type == "D":
        out[position] = "D+NMe" if current in METHYL else "D"
    else:
        raise ValueError(f"unknown edit type {edit_type!r}")

    return out


def build_pairs(project_root: Path) -> pd.DataFrame:
    """
    Load matched single-edit pairs with clamped, uncensored permeability.

    Returns one row per (reference, modified) pair with the columns the
    oracle needs: reference pattern, edited position, edit type, and the
    measured change in permeability.
    """

    root = Path(project_root)

    splits = pd.concat(
        [
            pd.read_csv(root / "data" / "splits" / f"{name}.csv")
            for name in ("train", "validation", "test")
        ],
        ignore_index=True,
    ).set_index("ID")

    pairs = pd.read_csv(
        root / "outputs" / "single_edit_pairs.csv", low_memory=False
    )

    pairs = pairs[pairs["clean_single_edit"] == True].copy()

    # The stored pair is unordered; reference_state says which side is the
    # unmodified one.
    reference_is_a = pairs["state_a"] == pairs["reference_state"]

    def first_id(column):
        return pairs[column].astype(str).str.split(",").str[0].astype(int)

    pairs["reference_id"] = np.where(
        reference_is_a, first_id("IDs_a"), first_id("IDs_b")
    )
    pairs["modified_id"] = np.where(
        reference_is_a, first_id("IDs_b"), first_id("IDs_a")
    )
    pairs["reference_pattern"] = np.where(
        reference_is_a, pairs["pattern_a"], pairs["pattern_b"]
    )

    permeability = splits["PAMPA"]
    censored = splits["censored"]

    pairs["delta"] = (
        permeability.reindex(pairs["modified_id"]).to_numpy()
        - permeability.reindex(pairs["reference_id"]).to_numpy()
    )

    pairs["censored"] = (
        censored.reindex(pairs["reference_id"]).to_numpy()
        | censored.reindex(pairs["modified_id"]).to_numpy()
    )

    pairs = pairs[pairs["delta"].notna() & ~pairs["censored"]]

    return pairs.reset_index(drop=True)


@dataclass
class DeltaOracle:
    """Predicts the permeability change caused by a single backbone edit."""

    model: GradientBoostingRegressor = field(
        default_factory=lambda: GradientBoostingRegressor(random_state=0)
    )

    def _design(self, pairs: pd.DataFrame):

        rows, keep = [], []

        for index, row in enumerate(pairs.itertuples(index=False)):

            pattern = parse_pattern(row.reference_pattern)

            if pattern is None:
                continue

            features = edit_features(
                pattern, int(row.position), row.edit_type
            )

            if features is None:
                continue

            rows.append(features)
            keep.append(index)

        return np.asarray(rows), np.asarray(keep)

    def fit(self, pairs: pd.DataFrame) -> "DeltaOracle":

        design, keep = self._design(pairs)

        self.model.fit(design, pairs["delta"].to_numpy()[keep])
        self.n_train_ = len(keep)

        return self

    def predict(self, pairs: pd.DataFrame) -> np.ndarray:

        design, keep = self._design(pairs)

        out = np.full(len(pairs), np.nan)
        out[keep] = self.model.predict(design)

        return out

    def score_edits(self, pattern, editable_positions=None):
        """
        Score every legal single edit from `pattern`.

        Returns a frame of (position, edit_type, resulting pattern,
        predicted delta), sorted best first. This is the interface the
        search consumes.
        """

        pattern = parse_pattern(pattern)

        if editable_positions is None:
            editable_positions = range(len(pattern))

        rows, design = [], []

        for position in editable_positions:
            for edit_type in ("NMe", "D"):

                current = pattern[position]

                # Skip edits that would not change anything.
                if edit_type == "NMe" and current in METHYL:
                    continue
                if edit_type == "D" and current in INVERTED:
                    continue

                features = edit_features(pattern, position, edit_type)

                if features is None:
                    continue

                design.append(features)
                rows.append(
                    {
                        "position": position,
                        "edit_type": edit_type,
                        "pattern": apply_edit(pattern, position, edit_type),
                    }
                )

        if not rows:
            return pd.DataFrame(
                columns=["position", "edit_type", "pattern", "predicted_delta"]
            )

        frame = pd.DataFrame(rows)
        frame["predicted_delta"] = self.model.predict(np.asarray(design))

        return frame.sort_values(
            "predicted_delta", ascending=False, ignore_index=True
        )

    @property
    def feature_importance(self) -> pd.Series:
        return pd.Series(
            self.model.feature_importances_, index=FEATURE_NAMES
        ).sort_values(ascending=False)
