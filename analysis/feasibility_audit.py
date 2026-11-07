"""
T4: dataset feasibility audit.

Four measurements that bound what any downstream result can mean. None of
them concern the quantity of data.

1. Assay reproducibility. Structures measured more than once give a noise
   floor. A predictor at that floor is not accurate, it is saturated, and
   ranking decisions below it are not defensible at any model quality.
2. Edit-space occupancy. How much of the k-edit neighbourhood around each
   parent was actually measured. Exhaustive search over measured patterns
   and exhaustive search over the reachable space are not the same thing.
3. Chemical reach. Proline cannot be N-methylated, so the usable search
   space is smaller than 4^n implies.
4. Ring-size concentration. Which ring sizes the matched-series evidence
   actually speaks about.

Reads the raw CycPeptMPDB table, the backbone annotations, the matched
series table and the prepared splits. Writes results/tables/
feasibility_audit.{json,csv} and prints the same figures.

Usage:
    python analysis/feasibility_audit.py
    python analysis/feasibility_audit.py --budget 3
"""

import argparse
import json
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

RAW_PEPTIDES = ROOT / "data" / "raw" / "CycPeptMPDB_Peptide_All.csv"
ANNOTATIONS = ROOT / "outputs" / "peptide_backbone_annotations.csv"
MATCHED_SERIES = ROOT / "outputs" / "matched_parent_series.csv"
SPLIT_DIR = ROOT / "data" / "splits"

TABLE_DIR = ROOT / "results" / "tables"

# Sentinels, as defined in analysis/prepare_oracle_data.py. A replicate
# estimate built on placeholders would measure the placeholder, not the
# assay, so both are excluded here rather than clamped.
SENTINEL_LOW = -10.0
SENTINEL_HIGH = -4.0

# Residues whose backbone nitrogen carries no proton and so cannot be
# N-methylated. Matched as a prefix so substituted prolines are caught.
NON_METHYLATABLE_PREFIXES = ("Pro",)


# --------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------

def load_raw() -> pd.DataFrame:

    raw = pd.read_csv(RAW_PEPTIDES, low_memory=False)
    raw.columns = [c.strip().lstrip("﻿") for c in raw.columns]

    # The raw table is wide and arrives fragmented; copying once keeps the
    # single added column from raising a PerformanceWarning.
    raw = raw.copy()

    raw["PAMPA_numeric"] = pd.to_numeric(raw["PAMPA"], errors="coerce")

    return raw


def clean_measurements(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Real PAMPA numbers only: no sentinel floor, no sentinel ceiling, no
    detection-limit bound.
    """

    return raw[
        raw["PAMPA_numeric"].notna()
        & (raw["PAMPA_numeric"] != SENTINEL_LOW)
        & (raw["PAMPA_numeric"] != SENTINEL_HIGH)
        & raw["Detection_Limit_1"].isna()
    ].copy()


def parse_signature(value):
    """
    sidechain_signature is written as a tuple literal. Parsed without
    eval so a malformed row degrades to None instead of executing.
    """

    if not isinstance(value, str):
        return None

    inner = value.strip()

    if not (inner.startswith("(") and inner.endswith(")")):
        return None

    inner = inner[1:-1]

    parts = [p.strip().strip("'\"") for p in inner.split("',")]
    parts = [p.strip().strip("'\"") for p in parts if p.strip()]

    return [p for p in parts if p]


# --------------------------------------------------------------------
# 1. Assay reproducibility
# --------------------------------------------------------------------

def replicate_spread(clean: pd.DataFrame, target_sd: float) -> dict:
    """
    Same structure measured more than once. Within-source groups isolate
    repeat measurement; the all-source groups additionally absorb
    between-laboratory disagreement, so both are reported.
    """

    results = {}

    for label, keys in [
        ("within_source", ["Structurally_Unique_ID", "Source"]),
        ("any_source", ["Structurally_Unique_ID"]),
    ]:

        grouped = clean.groupby(keys)["PAMPA_numeric"]

        frame = grouped.agg(["size", "min", "max", "std"])
        frame = frame[frame["size"] > 1]

        spread = frame["max"] - frame["min"]

        results[label] = {
            "n_groups": int(len(frame)),
            "n_records": int(frame["size"].sum()),
            "median_range": float(spread.median()),
            "mean_range": float(spread.mean()),
            "p90_range": float(spread.quantile(0.90)),
            "max_range": float(spread.max()),
            "median_sd": float(frame["std"].median()),
            "mean_sd": float(frame["std"].mean()),
        }

    sources = clean.groupby("Structurally_Unique_ID")["Source"].nunique()

    results["structures_in_multiple_sources"] = int((sources > 1).sum())
    results["uncensored_target_sd"] = float(target_sd)

    results["noise_share_of_signal"] = float(
        results["within_source"]["median_sd"] / target_sd
    )

    return results


# --------------------------------------------------------------------
# 2. Edit-space occupancy
# --------------------------------------------------------------------

def reachable_patterns(length: int, budget: int) -> int:
    """
    Patterns within `budget` Hamming edits of the parent, before
    residue-specific chemical constraints. Three alternatives per edited
    position, from M = {None, D, NMe, D+NMe}.
    """

    return sum(comb(length, i) * 3 ** i for i in range(budget + 1))


def edit_space_occupancy(series: pd.DataFrame, budget: int) -> dict:
    """
    The median parent is the honest summary here. The mean is dominated
    by one very large scaffold library and should not be quoted alone.
    """

    series = series.copy()

    series["reachable"] = series["ring_size"].apply(
        lambda n: reachable_patterns(int(n), budget)
    )

    usable = series[series["n_patterns"] >= 2].copy()
    usable["occupancy"] = usable["n_patterns"] / usable["reachable"]

    return {
        "edit_budget": int(budget),
        "n_parents_total": int(len(series)),
        "n_parents_ge2_patterns": int((series["n_patterns"] >= 2).sum()),
        "n_parents_ge3_patterns": int((series["n_patterns"] >= 3).sum()),
        "measured_patterns_median": float(usable["n_patterns"].median()),
        "measured_patterns_mean": float(usable["n_patterns"].mean()),
        "measured_patterns_max": int(usable["n_patterns"].max()),
        "reachable_median": float(usable["reachable"].median()),
        "occupancy_median_pct": float(100 * usable["occupancy"].median()),
        "occupancy_mean_pct": float(100 * usable["occupancy"].mean()),
        "ring_size_counts": {
            int(k): int(v)
            for k, v in usable["ring_size"].value_counts().sort_index().items()
        },
        "largest_parent_source": str(
            usable.loc[usable["n_patterns"].idxmax(), "Source"]
        ),
    }


# --------------------------------------------------------------------
# 3. Chemical reach
# --------------------------------------------------------------------

def editable_positions(annotations: pd.DataFrame) -> dict:
    """
    Fraction of backbone positions that an N-methylation edit can reach.
    """

    supported = annotations

    if "supported" in annotations.columns:
        supported = annotations[annotations["supported"] == True]  # noqa: E712

    parsed = supported["sidechain_signature"].apply(parse_signature)

    keep = parsed.notna()
    parsed = parsed[keep]

    lengths = parsed.apply(len)

    blocked = parsed.apply(
        lambda residues: sum(
            1
            for r in residues
            if str(r).startswith(NON_METHYLATABLE_PREFIXES)
        )
    )

    return {
        "n_peptides": int(len(parsed)),
        "ring_size_median": float(lengths.median()),
        "ring_size_mean": float(lengths.mean()),
        "blocked_per_peptide_mean": float(blocked.mean()),
        "peptides_with_blocked_pct": float(100 * (blocked > 0).mean()),
        "editable_positions_mean": float((lengths - blocked).mean()),
        "editable_fraction_pct": float(
            100 * (lengths - blocked).sum() / lengths.sum()
        ),
    }


# --------------------------------------------------------------------
# 4. Held-out resolving power
# --------------------------------------------------------------------

def heldout_concentration() -> dict:
    """
    How many independent parents the held-out sets actually contain, and
    how much of each set the largest parent accounts for. A mean computed
    across a handful of units, one of which dominates, carries almost no
    resolving power.
    """

    results = {}

    for name in ["validation", "test"]:

        frame = pd.read_csv(SPLIT_DIR / f"{name}.csv")

        counts = frame["sidechain_signature"].value_counts()

        results[name] = {
            "n_peptides": int(len(frame)),
            "n_parents": int(frame["sidechain_signature"].nunique()),
            "largest_parent_n": int(counts.iloc[0]),
            "largest_parent_share_pct": float(
                100 * counts.iloc[0] / len(frame)
            ),
            "parents_with_ge3_members": int((counts >= 3).sum()),
        }

    return results


# --------------------------------------------------------------------

def main(args) -> None:

    print("=== T4: dataset feasibility audit ===\n")

    raw = load_raw()
    clean = clean_measurements(raw)

    splits = pd.concat(
        [pd.read_csv(SPLIT_DIR / f"{n}.csv")
         for n in ["train", "validation", "test"]],
        ignore_index=True,
    )

    target_sd = splits.loc[~splits["censored"], "PAMPA"].std()

    annotations = pd.read_csv(ANNOTATIONS, low_memory=False)
    series = pd.read_csv(MATCHED_SERIES, low_memory=False)

    print(f"Raw records {len(raw):,} | clean PAMPA {len(clean):,} | "
          f"prepared peptides {len(splits):,}")

    # ----------------------------------------------------------
    # 1. Assay reproducibility
    # ----------------------------------------------------------

    noise = replicate_spread(clean, target_sd)

    print("\n--- 1. Assay reproducibility ---")

    for label in ["within_source", "any_source"]:
        block = noise[label]
        print(f"  {label:<14} groups {block['n_groups']:3,}  "
              f"records {block['n_records']:4,}  "
              f"median range {block['median_range']:.3f}  "
              f"median sd {block['median_sd']:.3f}  "
              f"max range {block['max_range']:.3f}")

    print(f"  structures measured in >1 source: "
          f"{noise['structures_in_multiple_sources']:,}")

    print(f"  uncensored target sd {target_sd:.3f}  ->  within-source "
          f"replicate sd is {100 * noise['noise_share_of_signal']:.0f}% "
          f"of it")

    # ----------------------------------------------------------
    # 2. Edit-space occupancy
    # ----------------------------------------------------------

    occupancy = edit_space_occupancy(series, args.budget)

    print(f"\n--- 2. Edit-space occupancy (k = {args.budget}) ---")

    print(f"  parents {occupancy['n_parents_total']:,}  "
          f"with >=2 patterns {occupancy['n_parents_ge2_patterns']}  "
          f"with >=3 {occupancy['n_parents_ge3_patterns']}")

    print(f"  measured patterns  median "
          f"{occupancy['measured_patterns_median']:.0f}  "
          f"mean {occupancy['measured_patterns_mean']:.1f}  "
          f"max {occupancy['measured_patterns_max']:,} "
          f"({occupancy['largest_parent_source']})")

    print(f"  reachable at k={args.budget}  median "
          f"{occupancy['reachable_median']:.0f}")

    print(f"  occupancy  median {occupancy['occupancy_median_pct']:.2f}%  "
          f"mean {occupancy['occupancy_mean_pct']:.2f}%")

    print("  ring sizes among usable parents: "
          + ", ".join(f"{k}:{v}"
                      for k, v in occupancy["ring_size_counts"].items()))

    # ----------------------------------------------------------
    # 3. Chemical reach
    # ----------------------------------------------------------

    reach = editable_positions(annotations)

    print("\n--- 3. Chemical reach ---")

    print(f"  peptides parsed {reach['n_peptides']:,}  "
          f"ring size median {reach['ring_size_median']:.0f}  "
          f"mean {reach['ring_size_mean']:.2f}")

    print(f"  non-methylatable residues per peptide "
          f"{reach['blocked_per_peptide_mean']:.2f}  "
          f"peptides with >=1: {reach['peptides_with_blocked_pct']:.1f}%")

    print(f"  editable positions mean "
          f"{reach['editable_positions_mean']:.2f} of "
          f"{reach['ring_size_mean']:.2f}  "
          f"({reach['editable_fraction_pct']:.1f}%)")

    # ----------------------------------------------------------
    # 4. Held-out resolving power
    # ----------------------------------------------------------

    heldout = heldout_concentration()

    print("\n--- 4. Held-out resolving power ---")

    for name, block in heldout.items():
        print(f"  {name:<11} {block['n_peptides']:4,} peptides  "
              f"{block['n_parents']:2,} parents  "
              f"largest parent {block['largest_parent_n']:4,} "
              f"({block['largest_parent_share_pct']:.1f}%)  "
              f"parents with >=3 members {block['parents_with_ge3_members']}")

    # ----------------------------------------------------------
    # Write
    # ----------------------------------------------------------

    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    audit = {
        "assay_reproducibility": noise,
        "edit_space_occupancy": occupancy,
        "chemical_reach": reach,
        "heldout_resolving_power": heldout,
    }

    with open(TABLE_DIR / "feasibility_audit.json", "w") as handle:
        json.dump(audit, handle, indent=2)

    flat = pd.DataFrame(
        [
            {"group": group, "quantity": key, "value": value}
            for group, block in audit.items()
            for key, value in block.items()
            if isinstance(value, (int, float, str))
        ]
    )

    flat.to_csv(TABLE_DIR / "feasibility_audit.csv", index=False)

    print(f"\nWrote {TABLE_DIR / 'feasibility_audit.json'}")
    print(f"Wrote {TABLE_DIR / 'feasibility_audit.csv'}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--budget",
        type=int,
        default=2,
        help="edit budget k used for the reachable-pattern count",
    )

    main(parser.parse_args())
