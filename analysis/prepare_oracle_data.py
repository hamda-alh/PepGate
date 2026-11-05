"""
T1: build leakage-safe oracle training data with documented censoring

Reads the raw CycPeptMPDB table and the parent-level split produced by
create_parent_split.py, attaches SMILES, flags censored measurements, clamps
the lower sentinel, and writes data/splits/{train,validation,test}.csv.
"""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

RAW_PEPTIDES = ROOT / "data" / "raw" / "CycPeptMPDB_Peptide_All.csv"
SPLIT_INPUT = ROOT / "outputs" / "peptides_with_split.csv"

SPLIT_DIR = ROOT / "data" / "splits"
TABLE_DIR = ROOT / "results" / "tables"

# CycPeptMPDB stores unmeasurable permeability as the sentinel -10.
# Our non-sentinel minimum is -8.4 and nothing falls between, so -8.5
# separates sentinels from real measurements without distorting any.
LOWER_CLAMP = -8.5

SENTINEL_LOW = -10.0
SENTINEL_HIGH = -4.0

KEEP_COLUMNS = [
    "ID",
    "Source",
    "SMILES",
    "HELM",
    "Sequence",
    "Monomer_Length",
    "sidechain_signature",
    "backbone_pattern",
    "PAMPA_raw",
    "PAMPA",
    "censored",
    "censored_low",
    "censored_high",
    "split",
]


def load_raw() -> pd.DataFrame:
    raw = pd.read_csv(RAW_PEPTIDES, low_memory=False)
    raw.columns = [c.strip().lstrip("﻿") for c in raw.columns]
    return raw


def flag_censoring(df: pd.DataFrame) -> pd.DataFrame:
    """
    Three kinds of censoring, all recorded rather than dropped.

    1. PAMPA == -10: CycPeptMPDB's floor for unmeasurable samples.
    2. Detection_Limit_1 present: the source paper reported a bound,
       not a point estimate (e.g. "< -6.5").
    3. PAMPA == -4 with a %T>100 note: too permeable to quantify.
    """

    detection_note = df["Detection_Limit_1"].notna()

    df["censored_low"] = (
        (df["PAMPA_raw"] == SENTINEL_LOW)
        | (detection_note & (df["PAMPA_raw"] != SENTINEL_HIGH))
    )

    df["censored_high"] = (
        (df["PAMPA_raw"] == SENTINEL_HIGH) & detection_note
    )

    df["censored"] = df["censored_low"] | df["censored_high"]

    return df


def main() -> None:

    print("=== T1: oracle dataset preparation ===\n")

    raw = load_raw()
    annotated = pd.read_csv(SPLIT_INPUT, low_memory=False)

    print(f"Annotated peptides carrying a split: {len(annotated):,}")

    merged = annotated.merge(
        raw[["ID", "SMILES", "Detection_Limit_1", "Detection_Limit_2"]],
        on="ID",
        how="left",
        validate="one_to_one",
    )

    missing_smiles = merged["SMILES"].isna().sum()

    if missing_smiles:
        raise ValueError(f"{missing_smiles} peptides have no SMILES.")

    print(f"SMILES attached for all {len(merged):,} peptides")

    merged = merged.rename(columns={"PAMPA": "PAMPA_raw"})
    merged = flag_censoring(merged)

    merged["PAMPA"] = merged["PAMPA_raw"].clip(lower=LOWER_CLAMP)

    n_clamped = (merged["PAMPA"] != merged["PAMPA_raw"]).sum()

    # ----------------------------------------------------------
    # Censoring report
    # ----------------------------------------------------------

    print("\n--- Censoring ---")

    print(f"PAMPA == {SENTINEL_LOW} (sentinel floor): "
          f"{(merged['PAMPA_raw'] == SENTINEL_LOW).sum():,}")

    print(f"Reported as a bound (Detection_Limit_1): "
          f"{merged['Detection_Limit_1'].notna().sum():,}")

    print(f"PAMPA == {SENTINEL_HIGH} with a detection note: "
          f"{merged['censored_high'].sum():,}")

    print(f"Flagged censored (any kind): {merged['censored'].sum():,} "
          f"({100 * merged['censored'].mean():.1f}%)")

    print(f"Clamped to {LOWER_CLAMP}: {n_clamped:,}")

    print("\n--- Target distribution ---")

    for label, series in [
        ("raw", merged["PAMPA_raw"]),
        ("clamped", merged["PAMPA"]),
        ("clamped, uncensored only", merged.loc[~merged["censored"], "PAMPA"]),
    ]:
        print(f"  {label:<28} n={len(series):5,}  "
              f"mean={series.mean():7.4f}  sd={series.std():6.4f}  "
              f"min={series.min():6.2f}  max={series.max():6.2f}")

    # ----------------------------------------------------------
    # Split report
    # ----------------------------------------------------------

    merged["is_townsend"] = merged["Source"] == "2020_Townsend"

    split_summary = (
        merged
        .groupby("split")
        .agg(
            n_peptides=("ID", "nunique"),
            n_parents=("sidechain_signature", "nunique"),
            n_sources=("Source", "nunique"),
            n_censored=("censored", "sum"),
            townsend_fraction=("is_townsend", "mean"),
            pampa_mean=("PAMPA", "mean"),
            pampa_sd=("PAMPA", "std"),
        )
        .round(4)
        .reset_index()
    )

    print("\n--- Splits ---")
    print(split_summary.to_string(index=False))

    # ----------------------------------------------------------
    # Write
    # ----------------------------------------------------------

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    for split_name in ["train", "validation", "test"]:

        subset = merged.loc[merged["split"] == split_name, KEEP_COLUMNS]

        path = SPLIT_DIR / f"{split_name}.csv"
        subset.to_csv(path, index=False)

        print(f"\nWrote {len(subset):,} rows -> {path}")

    censoring_summary = pd.DataFrame(
        [
            {"category": "sentinel_floor_-10",
             "n": int((merged["PAMPA_raw"] == SENTINEL_LOW).sum())},
            {"category": "reported_as_bound",
             "n": int(merged["Detection_Limit_1"].notna().sum())},
            {"category": "sentinel_ceiling_-4",
             "n": int(merged["censored_high"].sum())},
            {"category": "any_censored",
             "n": int(merged["censored"].sum())},
            {"category": "clamped_to_lower_bound", "n": int(n_clamped)},
            {"category": "total_peptides", "n": int(len(merged))},
        ]
    )

    censoring_summary.to_csv(
        TABLE_DIR / "censoring_summary.csv", index=False
    )

    split_summary.to_csv(
        TABLE_DIR / "split_summary.csv", index=False
    )

    print(f"\nWrote summaries -> {TABLE_DIR}")


if __name__ == "__main__":
    main()
