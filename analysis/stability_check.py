"""
Is within-edit-type position ranking a real effect?

The measured lift over random flipped sign between two reasonable analysis
choices (+0.086 on four parents, -0.053 on six). This reruns the same
measurement across every variation that could plausibly have been chosen —
parent-inclusion threshold, model seed, and a bootstrap over which parents
are held out — and reports the distribution of the answer.

An effect that is positive in most settings is weak but real. One that
straddles zero is noise, and nothing should be built on it.

Usage:
    python stability_check.py --project .
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))

from delta_oracle import DeltaOracle, build_pairs

THRESHOLDS = (20, 30, 40, 50, 100)
SEEDS = (0, 1, 2, 3, 4)


def lopo_lift(design, frame, edit_type, threshold, seed, train_all):
    """
    One leave-one-parent-out measurement.

    Returns per-parent rows of (top-1 rate, random rate), where candidates
    are restricted to a single edit type so the model cannot score by
    preferring methylation over inversion.
    """

    in_type = (frame["edit_type"] == edit_type).to_numpy()

    counts = frame[in_type].groupby("sidechain_signature").size()
    parents = counts[counts >= threshold].index.tolist()

    labels = frame["delta"].to_numpy()

    rows = []

    for held_out in parents:

        test = (frame["sidechain_signature"] == held_out).to_numpy()

        train = ~test if train_all else (~test) & in_type

        if train.sum() < 50:
            continue

        model = GradientBoostingRegressor(random_state=seed).fit(
            design[train], labels[train]
        )

        subset = frame[test & in_type].copy()

        if len(subset) < 10:
            continue

        subset["predicted"] = model.predict(design[test & in_type])

        hits = chance = groups = 0

        for _, group in subset.groupby("reference_pattern"):

            if len(group) < 3:
                continue

            groups += 1

            chosen = group["delta"].to_numpy()[
                group["predicted"].to_numpy().argmax()
            ]

            hits += chosen == group["delta"].max()
            chance += 1 / len(group)

        if groups >= 3:
            rows.append(
                {
                    "parent": held_out,
                    "top1": hits / groups,
                    "random": chance / groups,
                    "groups": groups,
                }
            )

    return pd.DataFrame(rows)


def main(args):

    pairs = build_pairs(args.project)

    oracle = DeltaOracle()
    design, keep = oracle._design(pairs)
    frame = pairs.iloc[keep].reset_index(drop=True)

    print(f"{len(frame):,} pairs\n")

    rng = np.random.default_rng(0)

    for train_all in (False, True):

        scope = (
            "trained on all edit types"
            if train_all
            else "trained on this edit type only"
        )

        print(f"=== {scope} ===")

        for edit_type in ("NMe", "D"):

            lifts, settings = [], []

            for threshold in THRESHOLDS:
                for seed in SEEDS:

                    result = lopo_lift(
                        design, frame, edit_type, threshold, seed, train_all
                    )

                    if len(result) < 2:
                        continue

                    lift = (result["top1"] - result["random"]).mean()

                    lifts.append(lift)
                    settings.append((threshold, seed, len(result)))

            if not lifts:
                print(f"  {edit_type}: too few parents at any threshold")
                continue

            lifts = np.asarray(lifts)

            # Bootstrap over parents at the middle threshold, to separate
            # "which parents" from "which hyperparameters".
            result = lopo_lift(design, frame, edit_type, 30, 0, train_all)

            boot = []

            if len(result) >= 3:
                per_parent = (result["top1"] - result["random"]).to_numpy()

                for _ in range(2000):
                    sample = rng.choice(
                        per_parent, size=len(per_parent), replace=True
                    )
                    boot.append(sample.mean())

                boot = np.asarray(boot)

            print(
                f"  {edit_type:<4} lift over {len(lifts)} settings: "
                f"mean {lifts.mean():+.4f}  sd {lifts.std():.4f}  "
                f"range [{lifts.min():+.4f}, {lifts.max():+.4f}]  "
                f"positive in {100 * (lifts > 0).mean():.0f}% of settings"
            )

            if len(boot):
                low, high = np.percentile(boot, [2.5, 97.5])
                print(
                    f"       bootstrap over {len(result)} parents: "
                    f"95% CI [{low:+.4f}, {high:+.4f}]  "
                    f"P(lift > 0) = {(boot > 0).mean():.2f}"
                )

        print()

    print("Reading: an effect positive in >90% of settings with a bootstrap")
    print("CI excluding zero is real. One straddling zero is not usable.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default=".")

    main(parser.parse_args())
