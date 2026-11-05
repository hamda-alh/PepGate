import ast
import random
from pathlib import Path

import pandas as pd

from optimizer import (
    STATE_FEATURES,
    hamming_distance,
)


ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    ROOT
    / "outputs"
    / "peptides_with_split.csv"
)

RUNS_OUTPUT = (
    ROOT
    / "outputs"
    / "validation_pampa_benchmark_runs.csv"
)

SUMMARY_OUTPUT = (
    ROOT
    / "outputs"
    / "validation_pampa_benchmark_summary.csv"
)

NON_TOWNSEND_SUMMARY_OUTPUT = (
    ROOT
    / "outputs"
    / "validation_pampa_benchmark_summary_no_townsend.csv"
)


# --------------------------------------------------
# Experiment settings
# --------------------------------------------------

EDIT_BUDGET = 2
ORACLE_CALL_BUDGET = 20
BEAM_WIDTH = 3
N_SEEDS = 100


def parse_tuple(value):
    if pd.isna(value):
        return None

    return tuple(ast.literal_eval(value))


def modification_load(pattern):
    """
    Count elementary backbone modifications.

    none     = 0
    D        = 1
    NMe      = 1
    D+NMe    = 2
    """

    return sum(
        len(STATE_FEATURES[state])
        for state in pattern
    )


def modified_positions(pattern):
    """
    Count positions differing from the unmodified state.
    """

    return sum(
        state != "none"
        for state in pattern
    )


def choose_start_pattern(pattern_scores):
    """
    Choose the least-modified measured pattern.

    PAMPA is NOT used to choose the starting parent.
    """

    patterns = list(pattern_scores)

    return min(
        patterns,
        key=lambda pattern: (
            modification_load(pattern),
            modified_positions(pattern),
            str(pattern),
        ),
    )


def measured_neighbors(
    pattern,
    candidate_patterns,
):
    """
    Return experimentally measured patterns that differ
    from `pattern` at exactly one backbone position.
    """

    return [
        candidate
        for candidate in candidate_patterns
        if (
            candidate != pattern
            and hamming_distance(
                pattern,
                candidate,
            ) == 1
        )
    ]


# --------------------------------------------------
# Random local search
# --------------------------------------------------

def random_measured_search(
    start,
    scores,
    call_budget,
    seed,
):
    """
    Randomly walk through measured one-position neighbors.
    """

    rng = random.Random(seed)

    patterns = set(scores)

    evaluated = {start}

    calls = 0

    while calls < call_budget:

        expandable = []

        for pattern in evaluated:

            unseen_neighbors = [
                candidate
                for candidate in measured_neighbors(
                    pattern,
                    patterns,
                )
                if candidate not in evaluated
            ]

            if unseen_neighbors:
                expandable.append(
                    (
                        pattern,
                        unseen_neighbors,
                    )
                )

        if not expandable:
            break

        _, neighbors = rng.choice(
            expandable
        )

        candidate = rng.choice(
            neighbors
        )

        evaluated.add(candidate)

        calls += 1

    best_pattern = max(
        evaluated,
        key=lambda pattern: scores[pattern],
    )

    return (
        best_pattern,
        scores[best_pattern],
        calls,
    )


# --------------------------------------------------
# Generate-and-rank
# --------------------------------------------------

def measured_generate_and_rank(
    start,
    scores,
    call_budget,
    seed,
):
    """
    Sample measured candidates globally once,
    score them, and return the best.
    """

    rng = random.Random(seed)

    candidates = [
        pattern
        for pattern in scores
        if pattern != start
    ]

    n = min(
        call_budget,
        len(candidates),
    )

    sampled = rng.sample(
        candidates,
        n,
    )

    evaluated = (
        [start]
        + sampled
    )

    best_pattern = max(
        evaluated,
        key=lambda pattern: scores[pattern],
    )

    return (
        best_pattern,
        scores[best_pattern],
        n,
    )


# --------------------------------------------------
# Guided measured search
# --------------------------------------------------

def guided_measured_search(
    start,
    scores,
    call_budget,
    beam_width,
    seed,
):
    """
    Beam-style guided optimization through the measured
    backbone landscape.
    """

    rng = random.Random(seed)

    patterns = set(scores)

    evaluated = {start}

    beam = [start]

    calls = 0

    while calls < call_budget:

        proposed = set()

        for pattern in beam:

            for candidate in measured_neighbors(
                pattern,
                patterns,
            ):

                if candidate not in evaluated:
                    proposed.add(candidate)

        if not proposed:
            break

        remaining = (
            call_budget - calls
        )

        proposed = list(proposed)

        if len(proposed) > remaining:
            proposed = rng.sample(
                proposed,
                remaining,
            )

        evaluated.update(
            proposed
        )

        calls += len(proposed)

        ranked = sorted(
            evaluated,
            key=lambda pattern: scores[pattern],
            reverse=True,
        )

        beam = ranked[
            :beam_width
        ]

    best_pattern = max(
        evaluated,
        key=lambda pattern: scores[pattern],
    )

    return (
        best_pattern,
        scores[best_pattern],
        calls,
    )


def main():

    print(
        "=== PepGate retrospective PAMPA benchmark ==="
    )

    print(
        "Split: validation"
    )

    print()

    # --------------------------------------------------
    # 1. Load validation peptides
    # --------------------------------------------------

    df = pd.read_csv(
        INPUT_PATH,
        low_memory=False,
    )

    df = df[
        df["split"] == "validation"
    ].copy()

    df["sidechain_signature"] = (
        df["sidechain_signature"]
        .apply(parse_tuple)
    )

    df["backbone_pattern"] = (
        df["backbone_pattern"]
        .apply(parse_tuple)
    )

    df["PAMPA_numeric"] = pd.to_numeric(
        df["PAMPA"],
        errors="coerce",
    )

    df = df[
        df["PAMPA_numeric"].notna()
    ].copy()

    # --------------------------------------------------
    # 2. Collapse replicate measurements
    # --------------------------------------------------

    pattern_level = (
        df.groupby(
            [
                "sidechain_signature",
                "Source",
                "backbone_pattern",
            ],
            as_index=False,
        )
        .agg(
            PAMPA=(
                "PAMPA_numeric",
                "median",
            ),
            n_measurements=(
                "ID",
                "nunique",
            ),
        )
    )

    # --------------------------------------------------
    # 3. Find matched source-specific series
    # --------------------------------------------------

    series_counts = (
        pattern_level
        .groupby(
            [
                "sidechain_signature",
                "Source",
            ],
            as_index=False,
        )
        .agg(
            n_patterns=(
                "backbone_pattern",
                "nunique",
            )
        )
    )

    qualified = series_counts[
        series_counts["n_patterns"] >= 3
    ].copy()

    # ----------------------------------------------
    # If one global parent appears in multiple
    # publications, use the source containing the
    # most measured backbone patterns.
    # ----------------------------------------------

    qualified = qualified.sort_values(
        [
            "sidechain_signature",
            "n_patterns",
            "Source",
        ],
        ascending=[
            True,
            False,
            True,
        ],
    )

    selected_series = (
        qualified
        .groupby(
            "sidechain_signature",
            as_index=False,
        )
        .first()
    )

    print(
        "Validation parents with >=3 measured "
        f"patterns: {len(selected_series)}"
    )

    print()

    # --------------------------------------------------
    # 4. Run retrospective optimization
    # --------------------------------------------------

    run_records = []

    skipped = []

    for _, selected in selected_series.iterrows():

        signature = selected[
            "sidechain_signature"
        ]

        source = selected[
            "Source"
        ]

        group = pattern_level[
            (
                pattern_level[
                    "sidechain_signature"
                ]
                == signature
            )
            & (
                pattern_level[
                    "Source"
                ]
                == source
            )
        ].copy()

        scores = {
            row[
                "backbone_pattern"
            ]: row["PAMPA"]

            for _, row
            in group.iterrows()
        }

        # ------------------------------------------
        # Starting parent = least-modified
        # experimentally measured pattern
        # ------------------------------------------

        start = choose_start_pattern(
            scores
        )

        start_score = scores[
            start
        ]

        # ------------------------------------------
        # Restrict evaluation to patterns within
        # PepGate's edit budget from this parent.
        # ------------------------------------------

        eligible_scores = {
            pattern: score
            for pattern, score
            in scores.items()
            if hamming_distance(
                pattern,
                start,
            )
            <= EDIT_BUDGET
        }

        if len(eligible_scores) < 3:

            skipped.append(
                (
                    signature,
                    source,
                    len(eligible_scores),
                )
            )

            continue

        optimal_pattern = max(
            eligible_scores,
            key=lambda pattern:
                eligible_scores[pattern],
        )

        optimal_score = (
            eligible_scores[
                optimal_pattern
            ]
        )

        candidate_budget = min(
            ORACLE_CALL_BUDGET,
            len(eligible_scores) - 1,
        )

        print(
            f"Parent {signature}"
        )

        print(
            f"  Source: {source}"
        )

        print(
            f"  Measured patterns: {len(scores)}"
        )

        print(
            f"  Reachable within k={EDIT_BUDGET}: "
            f"{len(eligible_scores)}"
        )

        print(
            f"  Start PAMPA: {start_score:.3f}"
        )

        print(
            f"  Best measured PAMPA: "
            f"{optimal_score:.3f}"
        )

        # ------------------------------------------
        # Repeated stochastic runs
        # ------------------------------------------

        for seed in range(
            N_SEEDS
        ):

            methods = {}

            methods[
                "random_search"
            ] = random_measured_search(
                start=start,
                scores=eligible_scores,
                call_budget=candidate_budget,
                seed=seed,
            )

            methods[
                "generate_and_rank"
            ] = measured_generate_and_rank(
                start=start,
                scores=eligible_scores,
                call_budget=candidate_budget,
                seed=seed,
            )

            methods[
                "guided_search"
            ] = guided_measured_search(
                start=start,
                scores=eligible_scores,
                call_budget=candidate_budget,
                beam_width=BEAM_WIDTH,
                seed=seed,
            )

            for (
                method,
                (
                    best_pattern,
                    best_score,
                    calls,
                ),
            ) in methods.items():

                regret = (
                    optimal_score
                    - best_score
                )

                delta = (
                    best_score
                    - start_score
                )

                found_optimum = (
                    abs(regret) < 1e-12
                )

                run_records.append(
                    {
                        "method": method,
                        "seed": seed,
                        "Source": source,
                        "sidechain_signature":
                            signature,
                        "start_pattern":
                            start,
                        "start_PAMPA":
                            start_score,
                        "best_pattern":
                            best_pattern,
                        "best_PAMPA":
                            best_score,
                        "optimal_pattern":
                            optimal_pattern,
                        "optimal_PAMPA":
                            optimal_score,
                        "delta_PAMPA":
                            delta,
                        "regret":
                            regret,
                        "found_optimum":
                            found_optimum,
                        "oracle_calls":
                            calls,
                        "n_measured_patterns":
                            len(scores),
                        "n_eligible_patterns":
                            len(
                                eligible_scores
                            ),
                    }
                )

        print()

    results = pd.DataFrame(
        run_records
    )

    if results.empty:
        print(
            "No validation series produced usable benchmark runs."
        )
        return

    # --------------------------------------------------
    # 5. Parent-level summary
    # --------------------------------------------------
    #
    # Average over seeds FIRST.
    #
    # This prevents a giant combinatorial library from
    # dominating the result simply because it contains
    # more variants.
    # --------------------------------------------------

    parent_summary = (
        results
        .groupby(
            [
                "method",
                "sidechain_signature",
                "Source",
            ],
            as_index=False,
        )
        .agg(
            mean_delta_PAMPA=(
                "delta_PAMPA",
                "mean",
            ),
            mean_regret=(
                "regret",
                "mean",
            ),
            median_regret=(
                "regret",
                "median",
            ),
            optimum_rate=(
                "found_optimum",
                "mean",
            ),
            mean_oracle_calls=(
                "oracle_calls",
                "mean",
            ),
        )
    )

    overall = (
        parent_summary
        .groupby(
            "method",
            as_index=False,
        )
        .agg(
            n_parents=(
                "sidechain_signature",
                "nunique",
            ),
            mean_parent_delta_PAMPA=(
                "mean_delta_PAMPA",
                "mean",
            ),
            mean_parent_regret=(
                "mean_regret",
                "mean",
            ),
            median_parent_regret=(
                "mean_regret",
                "median",
            ),
            mean_optimum_rate=(
                "optimum_rate",
                "mean",
            ),
            mean_oracle_calls=(
                "mean_oracle_calls",
                "mean",
            ),
        )
    )

    print()
    print(
        "=== Validation summary ==="
    )

    print()

    print(
        overall.to_string(
            index=False
        )
    )

    print()

    print(
        "Skipped series:",
        len(skipped),
    )

    for item in skipped:

        print(
            " ",
            item,
        )

    # --------------------------------------------------
    # Sensitivity analysis: exclude 2020_Townsend
    # --------------------------------------------------

    non_townsend = parent_summary[
        parent_summary["Source"] != "2020_Townsend"
    ].copy()

    non_townsend_overall = (
        non_townsend
        .groupby(
            "method",
            as_index=False,
        )
        .agg(
            n_parents=(
                "sidechain_signature",
                "nunique",
            ),
            mean_parent_delta_PAMPA=(
                "mean_delta_PAMPA",
                "mean",
            ),
            mean_parent_regret=(
                "mean_regret",
                "mean",
            ),
            median_parent_regret=(
                "mean_regret",
                "median",
            ),
            mean_optimum_rate=(
                "optimum_rate",
                "mean",
            ),
            mean_oracle_calls=(
                "mean_oracle_calls",
                "mean",
            ),
        )
    )

    print()
    print(
        "=== Validation summary excluding 2020_Townsend ==="
    )
    print()

    print(
        non_townsend_overall.to_string(
            index=False
        )
    )

    # --------------------------------------------------
    # 6. Save outputs
    # --------------------------------------------------

    RUNS_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        RUNS_OUTPUT,
        index=False,
    )

    overall.to_csv(
        SUMMARY_OUTPUT,
        index=False,
    )

    non_townsend_overall.to_csv(
        NON_TOWNSEND_SUMMARY_OUTPUT,
        index=False,
    )

    print()

    print(
        f"Runs saved to: {RUNS_OUTPUT}"
    )

    print(
        f"Summary saved to: {SUMMARY_OUTPUT}"
    )

    print(
        "Non-Townsend summary saved to: "
        f"{NON_TOWNSEND_SUMMARY_OUTPUT}"
    )


if __name__ == "__main__":
    main()
