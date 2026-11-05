from pathlib import Path

import pandas as pd

from backbone import (
    annotate_sequence,
    build_monomer_map,
    canonicalize_ring,
    load_monomer_table,
)


ROOT = Path(__file__).resolve().parents[1]

PEPTIDE_PATH = (
    ROOT
    / "data"
    / "raw"
    / "CycPeptMPDB_Peptide_All.csv"
)

MONOMER_PATH = (
    ROOT
    / "data"
    / "raw"
    / "CycPeptMPDB_Monomer_All.csv"
)

OUTPUT_DIR = ROOT / "outputs"


def main():

    # --------------------------------------------------
    # 1. Load data
    # --------------------------------------------------

    peptides = pd.read_csv(
        PEPTIDE_PATH,
        low_memory=False,
    )

    monomers = load_monomer_table(
        MONOMER_PATH
    )

    monomer_map = build_monomer_map(
        monomers
    )

    print("=== PepGate backbone feasibility audit ===")
    print()

    print(
        f"Total peptides in CycPeptMPDB: "
        f"{len(peptides):,}"
    )

    # --------------------------------------------------
    # 2. Keep cyclic peptides with PAMPA measurements
    # --------------------------------------------------

    peptides = peptides[
        (peptides["Molecule_Shape"] == "Circle")
        & (peptides["PAMPA"].notna())
    ].copy()

    print(
        f"Cyclic peptides with PAMPA: "
        f"{len(peptides):,}"
    )

    # --------------------------------------------------
    # 3. Parse backbone / side-chain information
    # --------------------------------------------------

    records = []

    for _, row in peptides.iterrows():

        annotation = annotate_sequence(
            row["Sequence"],
            monomer_map,
        )

        record = {
            "ID": row["ID"],
            "Source": row["Source"],
            "PAMPA": row["PAMPA"],
            "Sequence": row["Sequence"],
            "HELM": row["HELM"],
            "Monomer_Length": row["Monomer_Length"],
            "supported": annotation["supported"],
            "reason": annotation["reason"],
        }

        if annotation["supported"]:

            sidechains, backbone = canonicalize_ring(
                annotation["sidechains"],
                annotation["backbone"],
            )

            record["sidechain_signature"] = sidechains
            record["backbone_pattern"] = backbone

        else:

            record["sidechain_signature"] = None
            record["backbone_pattern"] = None

        records.append(record)

    annotated = pd.DataFrame(records)

    # --------------------------------------------------
    # 4. How much of the dataset fits PepGate's edit space?
    # --------------------------------------------------

    supported = annotated[
        annotated["supported"]
    ].copy()

    print(
        f"Peptides supported by PepGate edit space: "
        f"{len(supported):,}"
    )

    print(
        f"Excluded peptides: "
        f"{len(annotated) - len(supported):,}"
    )

    # --------------------------------------------------
    # 5. Group peptides into matched parent series
    # --------------------------------------------------
    #
    # We use Source + side-chain signature so that
    # measurements from different publications are not
    # automatically treated as one experimental series.
    # --------------------------------------------------

    series = (
        supported
        .groupby(
            [
                "Source",
                "sidechain_signature",
            ]
        )
        .agg(
            n_peptides=("ID", "nunique"),
            n_patterns=(
                "backbone_pattern",
                "nunique",
            ),
            ring_size=(
                "Monomer_Length",
                "first",
            ),
        )
        .reset_index()
    )

    print()
    print("--- Matched-series results ---")

    print(
        f"Unique side-chain parent series: "
        f"{len(series):,}"
    )

    # --------------------------------------------------
    # 6. Count parents with multiple backbone patterns
    # --------------------------------------------------

    for minimum in [2, 3, 5, 10]:

        count = (
            series["n_patterns"] >= minimum
        ).sum()

        print(
            f"Parents with >= {minimum} "
            f"backbone patterns: {count:,}"
        )

    # --------------------------------------------------
    # 7. Count peptides contained in useful series
    # --------------------------------------------------

    matched_2 = series[
        series["n_patterns"] >= 2
    ][
        [
            "Source",
            "sidechain_signature",
        ]
    ]

    matched_3 = series[
        series["n_patterns"] >= 3
    ][
        [
            "Source",
            "sidechain_signature",
        ]
    ]

    peptides_in_2 = supported.merge(
        matched_2,
        on=[
            "Source",
            "sidechain_signature",
        ],
        how="inner",
    )

    peptides_in_3 = supported.merge(
        matched_3,
        on=[
            "Source",
            "sidechain_signature",
        ],
        how="inner",
    )

    print()
    print(
        f"Peptides in >=2-pattern series: "
        f"{peptides_in_2['ID'].nunique():,}"
    )

    print(
        f"Peptides in >=3-pattern series: "
        f"{peptides_in_3['ID'].nunique():,}"
    )

    # --------------------------------------------------
    # 8. Distribution of patterns per parent
    # --------------------------------------------------

    matched_series = series[
        series["n_patterns"] >= 2
    ]

    if len(matched_series) > 0:

        print()

        print(
            "Median patterns per matched parent:",
            matched_series["n_patterns"].median(),
        )

        print(
            "Maximum patterns for one parent:",
            matched_series["n_patterns"].max(),
        )

    # --------------------------------------------------
    # 9. Show most common exclusion reasons
    # --------------------------------------------------

    excluded = annotated[
        ~annotated["supported"]
    ]

    print()
    print("--- Most common exclusion reasons ---")

    print(
        excluded["reason"]
        .value_counts()
        .head(10)
    )

    # --------------------------------------------------
    # 10. Save results
    # --------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    annotated.to_csv(
        OUTPUT_DIR
        / "peptide_backbone_annotations.csv",
        index=False,
    )

    series.to_csv(
        OUTPUT_DIR
        / "matched_parent_series.csv",
        index=False,
    )

    print()
    print("Results written to outputs/")


if __name__ == "__main__":
    main()