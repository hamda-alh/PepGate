import ast
from itertools import combinations
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    ROOT
    / "outputs"
    / "peptide_backbone_annotations.csv"
)

OUTPUT_PATH = (
    ROOT
    / "outputs"
    / "single_edit_pairs.csv"
)


# Each backbone state can be represented by the
# elementary modifications it contains.
STATE_FEATURES = {
    "none": frozenset(),
    "D": frozenset({"D"}),
    "NMe": frozenset({"NMe"}),
    "D+NMe": frozenset({"D", "NMe"}),
}


def parse_tuple(value):
    """
    Convert tuple strings saved in the CSV back into tuples.
    """

    if pd.isna(value):
        return None

    parsed = ast.literal_eval(value)

    return tuple(parsed)


def differing_positions(pattern_a, pattern_b):
    """
    Return positions where two backbone patterns differ.
    """

    if len(pattern_a) != len(pattern_b):
        return []

    return [
        i
        for i, (a, b) in enumerate(
            zip(pattern_a, pattern_b)
        )
        if a != b
    ]


def orient_single_edit(state_a, state_b):
    """
    Determine whether changing state_a to state_b represents
    exactly one elementary PepGate edit.

    Examples:
        none -> NMe
        none -> D
        D -> D+NMe
        NMe -> D+NMe

    Returns
    -------
    reference_state
    modified_state
    edit_type

    or None if the state change requires two elementary edits.
    """

    features_a = STATE_FEATURES[state_a]
    features_b = STATE_FEATURES[state_b]

    # A -> B adds exactly one modification.
    if (
        features_a < features_b
        and len(features_b - features_a) == 1
    ):
        added = next(
            iter(features_b - features_a)
        )

        return state_a, state_b, added

    # B -> A adds exactly one modification.
    if (
        features_b < features_a
        and len(features_a - features_b) == 1
    ):
        added = next(
            iter(features_a - features_b)
        )

        return state_b, state_a, added

    # Examples:
    # none <-> D+NMe
    # D <-> NMe
    #
    # These differ at one position but require
    # two elementary changes.
    return None


def main():

    print(
        "=== PepGate single-edit pair audit ==="
    )
    print()

    df = pd.read_csv(
        INPUT_PATH,
        low_memory=False,
    )

    # -----------------------------------------
    # 1. Keep supported peptides
    # -----------------------------------------

    df = df[
        df["backbone_pattern"].notna()
        & df["sidechain_signature"].notna()
    ].copy()

    # Convert saved tuple strings back into tuples.
    df["backbone_pattern"] = (
        df["backbone_pattern"]
        .apply(parse_tuple)
    )

    df["sidechain_signature"] = (
        df["sidechain_signature"]
        .apply(parse_tuple)
    )

    # Make sure PAMPA is numeric.
    df["PAMPA_numeric"] = pd.to_numeric(
        df["PAMPA"],
        errors="coerce",
    )

    before = len(df)

    df = df[
        df["PAMPA_numeric"].notna()
    ].copy()

    print(
        f"Supported peptides with numeric PAMPA: "
        f"{len(df):,}"
    )

    print(
        f"Removed because PAMPA was non-numeric: "
        f"{before - len(df):,}"
    )

    # -----------------------------------------
    # 2. Collapse duplicate measurements of the
    #    same backbone pattern within one parent
    # -----------------------------------------

    pattern_level = (
        df.groupby(
            [
                "Source",
                "sidechain_signature",
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
            peptide_ids=(
                "ID",
                lambda x: ",".join(
                    map(str, sorted(set(x)))
                ),
            ),
        )
    )

    print(
        f"Unique measured backbone patterns: "
        f"{len(pattern_level):,}"
    )

    # -----------------------------------------
    # 3. Compare patterns inside each parent
    # -----------------------------------------

    pair_records = []

    grouped = pattern_level.groupby(
        [
            "Source",
            "sidechain_signature",
        ]
    )

    for (
        source,
        sidechain_signature,
    ), group in grouped:

        records = group.to_dict("records")

        if len(records) < 2:
            continue

        for row_a, row_b in combinations(
            records,
            2,
        ):

            pattern_a = row_a[
                "backbone_pattern"
            ]

            pattern_b = row_b[
                "backbone_pattern"
            ]

            differences = differing_positions(
                pattern_a,
                pattern_b,
            )

            # We only care about pairs differing
            # at one backbone position.
            if len(differences) != 1:
                continue

            position = differences[0]

            state_a = pattern_a[position]
            state_b = pattern_b[position]

            orientation = orient_single_edit(
                state_a,
                state_b,
            )

            record = {
                "Source": source,
                "sidechain_signature":
                    sidechain_signature,
                "position": position,
                "pattern_a": pattern_a,
                "pattern_b": pattern_b,
                "state_a": state_a,
                "state_b": state_b,
                "PAMPA_a": row_a["PAMPA"],
                "PAMPA_b": row_b["PAMPA"],
                "IDs_a": row_a[
                    "peptide_ids"
                ],
                "IDs_b": row_b[
                    "peptide_ids"
                ],
                "one_position_difference":
                    True,
            }

            if orientation is not None:

                (
                    reference_state,
                    modified_state,
                    edit_type,
                ) = orientation

                # Determine which peptide is the
                # reference and which is modified.
                if state_a == reference_state:

                    reference_pampa = (
                        row_a["PAMPA"]
                    )

                    modified_pampa = (
                        row_b["PAMPA"]
                    )

                else:

                    reference_pampa = (
                        row_b["PAMPA"]
                    )

                    modified_pampa = (
                        row_a["PAMPA"]
                    )

                delta_pampa = (
                    modified_pampa
                    - reference_pampa
                )

                record.update(
                    {
                        "clean_single_edit":
                            True,
                        "reference_state":
                            reference_state,
                        "modified_state":
                            modified_state,
                        "edit_type":
                            edit_type,
                        "reference_PAMPA":
                            reference_pampa,
                        "modified_PAMPA":
                            modified_pampa,
                        "delta_PAMPA":
                            delta_pampa,
                    }
                )

            else:

                record.update(
                    {
                        "clean_single_edit":
                            False,
                        "reference_state":
                            None,
                        "modified_state":
                            None,
                        "edit_type":
                            None,
                        "reference_PAMPA":
                            None,
                        "modified_PAMPA":
                            None,
                        "delta_PAMPA":
                            None,
                    }
                )

            pair_records.append(record)

    pairs = pd.DataFrame(pair_records)

    # -----------------------------------------
    # 4. Summary
    # -----------------------------------------

    print()
    print("--- One-position pair results ---")

    print(
        f"Pairs differing at exactly one position: "
        f"{len(pairs):,}"
    )

    clean = pairs[
        pairs["clean_single_edit"]
    ].copy()

    print(
        f"Clean single-edit pairs: "
        f"{len(clean):,}"
    )

    print()
    print("--- Edit types ---")

    transition_counts = (
        clean.groupby(
            [
                "reference_state",
                "modified_state",
            ]
        )
        .size()
        .sort_values(
            ascending=False
        )
    )

    print(
        transition_counts.to_string()
    )

    print()
    print("--- Elementary modifications ---")

    print(
        clean[
            "edit_type"
        ]
        .value_counts()
        .to_string()
    )

    # -----------------------------------------
    # 5. PAMPA effect
    # -----------------------------------------

    print()
    print("--- PAMPA changes ---")

    improved = (
        clean["delta_PAMPA"] > 0
    ).sum()

    decreased = (
        clean["delta_PAMPA"] < 0
    ).sum()

    unchanged = (
        clean["delta_PAMPA"] == 0
    ).sum()

    print(
        f"Positive delta PAMPA: {improved:,}"
    )

    print(
        f"Negative delta PAMPA: {decreased:,}"
    )

    print(
        f"No change: {unchanged:,}"
    )

    print()

    print(
        "Median delta PAMPA:",
        clean["delta_PAMPA"].median(),
    )

    # -----------------------------------------
    # 6. Number of independent parent series
    # -----------------------------------------

    n_parents = (
        clean[
            [
                "Source",
                "sidechain_signature",
            ]
        ]
        .drop_duplicates()
        .shape[0]
    )

    print()

    print(
        f"Parents containing clean "
        f"single-edit pairs: {n_parents:,}"
    )

    # -----------------------------------------
    # 7. Save
    # -----------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pairs.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print(
        f"Saved to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()