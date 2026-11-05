import ast
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

ANNOTATION_PATH = (
    ROOT
    / "outputs"
    / "peptide_backbone_annotations.csv"
)

MATCHED_PATH = (
    ROOT
    / "outputs"
    / "matched_parent_series.csv"
)

OUTPUT_PATH = (
    ROOT
    / "outputs"
    / "parent_split.csv"
)

SEED = 42


def parse_tuple(value):
    if pd.isna(value):
        return None

    return tuple(ast.literal_eval(value))


def main():

    print("=== PepGate parent-level split ===")
    print()

    # --------------------------------------------------
    # 1. Load annotated peptides
    # --------------------------------------------------

    peptides = pd.read_csv(
        ANNOTATION_PATH,
        low_memory=False,
    )

    matched = pd.read_csv(
        MATCHED_PATH,
        low_memory=False,
    )

    # Keep only supported peptides.
    peptides = peptides[
        peptides["sidechain_signature"].notna()
        & peptides["backbone_pattern"].notna()
    ].copy()

    peptides["sidechain_signature"] = (
        peptides["sidechain_signature"]
        .apply(parse_tuple)
    )

    peptides["backbone_pattern"] = (
        peptides["backbone_pattern"]
        .apply(parse_tuple)
    )

    matched["sidechain_signature"] = (
        matched["sidechain_signature"]
        .apply(parse_tuple)
    )

    # --------------------------------------------------
    # 2. Build GLOBAL parent IDs
    # --------------------------------------------------
    #
    # Important:
    # Split by side-chain signature alone.
    #
    # If the same scaffold appears in two papers,
    # it must still stay in one split.
    # --------------------------------------------------

    parent_stats = (
        peptides
        .groupby(
            "sidechain_signature",
            as_index=False,
        )
        .agg(
            n_peptides=("ID", "nunique"),
            n_patterns=(
                "backbone_pattern",
                "nunique",
            ),
            n_sources=(
                "Source",
                "nunique",
            ),
        )
    )

    # --------------------------------------------------
    # 3. Identify parents suitable for evaluation
    # --------------------------------------------------
    #
    # Use the SAME-SOURCE matched-series audit to decide
    # whether a parent has >=3 measured backbone patterns.
    # --------------------------------------------------

    eval_series = matched[
        matched["n_patterns"] >= 3
    ].copy()

    eval_parent_signatures = set(
        eval_series[
            "sidechain_signature"
        ]
    )

    parent_stats[
        "evaluation_eligible"
    ] = parent_stats[
        "sidechain_signature"
    ].isin(
        eval_parent_signatures
    )

    # Does the parent appear in Townsend?
    townsend_signatures = set(
        peptides.loc[
            peptides["Source"]
            == "2020_Townsend",
            "sidechain_signature",
        ]
    )

    parent_stats[
        "has_townsend"
    ] = parent_stats[
        "sidechain_signature"
    ].isin(
        townsend_signatures
    )

    eligible = parent_stats[
        parent_stats[
            "evaluation_eligible"
        ]
    ].copy()

    print(
        f"Total global parent scaffolds: "
        f"{len(parent_stats):,}"
    )

    print(
        f"Parents eligible for matched-pattern "
        f"evaluation: {len(eligible):,}"
    )

    # --------------------------------------------------
    # 4. Split matched parents into
    #    train / validation / test
    # --------------------------------------------------

    rng = np.random.default_rng(SEED)

    townsend = eligible[
        eligible["has_townsend"]
    ].copy()

    non_townsend = eligible[
        ~eligible["has_townsend"]
    ].copy()

    # Shuffle deterministically.
    townsend = townsend.iloc[
        rng.permutation(len(townsend))
    ]

    non_townsend = non_townsend.iloc[
        rng.permutation(len(non_townsend))
    ]

    # We want roughly:
    #
    # 60% matched parents -> training
    # 20% -> validation
    # 20% -> test
    #
    # But assign Townsend and non-Townsend separately
    # so one split does not accidentally contain all
    # of the large Townsend libraries.

    def split_group(group):

        n = len(group)

        if n == 0:
            return [], [], []

        n_test = max(
            1,
            round(n * 0.20),
        )

        n_val = max(
            1,
            round(n * 0.20),
        )

        # Make sure train is not empty.
        if n_test + n_val >= n:
            n_test = 1
            n_val = 1

        test = group.iloc[
            :n_test
        ]

        val = group.iloc[
            n_test:n_test + n_val
        ]

        train = group.iloc[
            n_test + n_val:
        ]

        return (
            train["sidechain_signature"].tolist(),
            val["sidechain_signature"].tolist(),
            test["sidechain_signature"].tolist(),
        )

    (
        train_townsend,
        val_townsend,
        test_townsend,
    ) = split_group(townsend)

    (
        train_other,
        val_other,
        test_other,
    ) = split_group(non_townsend)

    train_eval = set(
        train_townsend
        + train_other
    )

    val_parents = set(
        val_townsend
        + val_other
    )

    test_parents = set(
        test_townsend
        + test_other
    )

    # --------------------------------------------------
    # 5. All non-evaluation parents go to training
    # --------------------------------------------------

    all_parents = set(
        parent_stats[
            "sidechain_signature"
        ]
    )

    non_eval = (
        all_parents
        - eval_parent_signatures
    )

    train_parents = (
        train_eval
        | non_eval
    )

    # --------------------------------------------------
    # 6. Verify zero leakage
    # --------------------------------------------------

    assert train_parents.isdisjoint(
        val_parents
    )

    assert train_parents.isdisjoint(
        test_parents
    )

    assert val_parents.isdisjoint(
        test_parents
    )

    # --------------------------------------------------
    # 7. Assign split labels
    # --------------------------------------------------

    def assign_split(signature):

        if signature in test_parents:
            return "test"

        if signature in val_parents:
            return "validation"

        return "train"

    parent_stats["split"] = (
        parent_stats[
            "sidechain_signature"
        ]
        .apply(assign_split)
    )

    # --------------------------------------------------
    # 8. Print parent counts
    # --------------------------------------------------

    print()
    print("--- Parent counts ---")

    print(
        parent_stats[
            "split"
        ].value_counts()
    )

    print()
    print(
        "--- Evaluation-eligible parents ---"
    )

    print(
        parent_stats[
            parent_stats[
                "evaluation_eligible"
            ]
        ]
        ["split"]
        .value_counts()
    )

    # --------------------------------------------------
    # 9. Count peptides in each split
    # --------------------------------------------------

    peptides = peptides.merge(
        parent_stats[
            [
                "sidechain_signature",
                "split",
            ]
        ],
        on="sidechain_signature",
        how="left",
    )

    print()
    print("--- Peptide counts ---")

    print(
        peptides[
            "split"
        ].value_counts()
    )

    # --------------------------------------------------
    # 10. Townsend distribution
    # --------------------------------------------------

    print()
    print(
        "--- Townsend evaluation parents ---"
    )

    print(
        parent_stats[
            parent_stats[
                "evaluation_eligible"
            ]
            & parent_stats[
                "has_townsend"
            ]
        ]
        ["split"]
        .value_counts()
    )

    # --------------------------------------------------
    # 11. Save
    # --------------------------------------------------

    parent_stats.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    peptide_output = (
        ROOT
        / "outputs"
        / "peptides_with_split.csv"
    )

    peptides.to_csv(
        peptide_output,
        index=False,
    )

    print()
    print(
        f"Saved parent split to: "
        f"{OUTPUT_PATH}"
    )

    print(
        f"Saved peptide split to: "
        f"{peptide_output}"
    )


if __name__ == "__main__":
    main()