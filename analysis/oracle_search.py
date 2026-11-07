"""
Run PepGate's guided search with the learned delta oracle in place of
measured permeability.

The optimizer wants score_fn(pattern) -> float, but the oracle predicts the
effect of a single edit. We bridge the two by walking from the parent to the
candidate one position at a time, accumulating predicted deltas and updating
the backbone context at each step, so the oracle sees the pattern as it
actually looks when each edit is applied.

Usage:
    python oracle_search.py --project /path/to/PepGate
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from delta_oracle import (
    INVERTED,
    METHYL,
    DeltaOracle,
    build_pairs,
    edit_features,
    parse_pattern,
)


def edit_between(before: str, after: str) -> str | None:
    """Which single edit turns state `before` into state `after`."""

    if before == after:
        return None

    gained_methyl = after in METHYL and before not in METHYL
    gained_inversion = after in INVERTED and before not in INVERTED

    if gained_methyl and not gained_inversion:
        return "NMe"

    if gained_inversion and not gained_methyl:
        return "D"

    # Anything else (losing a modification, or gaining both at once) is not
    # a single elementary edit.
    return None


def make_score_fn(oracle: DeltaOracle, parent: tuple[str, ...]):
    """
    Build a pattern scorer from the delta oracle.

    Score is the predicted permeability change relative to the parent,
    summed over the edits that separate them.
    """

    parent = list(parent)

    def score(pattern) -> float:

        pattern = list(pattern)

        positions = [
            i for i in range(len(parent)) if parent[i] != pattern[i]
        ]

        if not positions:
            return 0.0

        running = list(parent)
        total = 0.0

        for position in sorted(positions):

            edit = edit_between(running[position], pattern[position])

            if edit is None:
                # Not reachable by elementary edits; penalise so the search
                # does not chase it.
                return -np.inf

            features = edit_features(running, position, edit)

            if features is None:
                return -np.inf

            total += float(
                oracle.model.predict(np.asarray([features]))[0]
            )

            running[position] = pattern[position]

        return total

    return score


def measured_landscape(pairs: pd.DataFrame, parent_key) -> dict:
    """
    Every measured pattern for one parent series, as pattern -> permeability.

    Built from both sides of each matched pair, so it covers the reference
    patterns as well as the edited ones.
    """

    subset = pairs[pairs["sidechain_signature"] == parent_key]

    landscape = {}

    for row in subset.itertuples(index=False):

        reference = parse_pattern(row.reference_pattern)

        if reference is None:
            continue

        modified = list(reference)
        position = int(row.position)

        if not 0 <= position < len(modified):
            continue

        before = modified[position]

        if row.edit_type == "NMe":
            modified[position] = "D+NMe" if before in INVERTED else "NMe"
        else:
            modified[position] = "D+NMe" if before in METHYL else "D"

        landscape.setdefault(tuple(reference), []).append(
            row.reference_PAMPA
        )
        landscape.setdefault(tuple(modified), []).append(row.modified_PAMPA)

    return {k: float(np.median(v)) for k, v in landscape.items()}


def main(args):

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(args.project) / "src"))

    from optimizer import guided_search, random_search

    pairs = build_pairs(args.project)

    parents = [
        key
        for key, group in pairs.groupby("sidechain_signature")
        if len(group) >= args.min_pairs
    ]

    print(f"{len(pairs)} clean pairs, {len(parents)} parents with "
          f">= {args.min_pairs} pairs\n")

    results = []

    for held_out in parents:

        train = pairs[pairs["sidechain_signature"] != held_out]

        oracle = DeltaOracle().fit(train)

        landscape = measured_landscape(pairs, held_out)

        if len(landscape) < args.min_landscape:
            continue

        # Start from the least-modified measured pattern, never the one with
        # the worst permeability.
        start = min(
            landscape,
            key=lambda p: sum(s != "none" for s in p),
        )

        reachable = {
            pattern: value
            for pattern, value in landscape.items()
            if sum(a != b for a, b in zip(start, pattern)) <= args.budget
        }

        if len(reachable) < 3:
            continue

        best_possible = max(reachable.values())

        # The oracle ranks the measured candidates. Searching the full
        # edit space is not evaluable here: almost every pattern the
        # search can reach has never been measured, so there is nothing
        # to score it against. Restricting to the measured landscape is
        # the same constraint the measured-PAMPA benchmark works under.
        score = make_score_fn(oracle, start)

        candidates = [p for p in reachable if p != tuple(start)]

        ranked = sorted(candidates, key=score, reverse=True)

        proposal = ranked[0]
        achieved = reachable[proposal]

        # Where would the true best sit in the oracle's ranking?
        true_best = max(candidates, key=lambda p: reachable[p])
        rank_of_best = ranked.index(true_best) + 1

        results.append(
            {
                "parent": str(held_out)[:28],
                "n_measured": len(reachable),
                "start": round(reachable[tuple(start)], 3),
                "oracle_pick": round(achieved, 3),
                "best_possible": round(best_possible, 3),
                "gain": round(achieved - reachable[tuple(start)], 3),
                "max_gain": round(
                    best_possible - reachable[tuple(start)], 3
                ),
                "regret": round(best_possible - achieved, 3),
                "rank_of_best": rank_of_best,
                "n_cand": len(candidates),
            }
        )

    frame = pd.DataFrame(results)

    print("=== PepGate with the learned delta oracle, leave-one-parent-out ===")
    print(frame.to_string(index=False))

    if len(frame):
        expected = (frame["n_cand"] + 1) / 2
        print(f"\nmean regret {frame['regret'].mean():.3f}  "
              f"gain captured "
              f"{100 * frame['gain'].sum() / frame['max_gain'].sum():.1f}%  "
              f"optimum found {(frame['regret'] == 0).mean():.1%}")
        print(f"mean rank of the true best: {frame['rank_of_best'].mean():.1f} "
              f"of {frame['n_cand'].mean():.0f} candidates "
              f"(random would be {expected.mean():.1f})")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--budget", type=int, default=2)
    parser.add_argument("--calls", type=int, default=20)
    parser.add_argument("--min-pairs", type=int, default=40)
    parser.add_argument("--min-landscape", type=int, default=5)

    main(parser.parse_args())
