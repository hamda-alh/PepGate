"""
Does 3D structure explain the stereoinversion effects that backbone context
cannot?

Sequence-level features predict N-methylation effects (R^2 ~ 0.11) but say
nothing about L->D inversion (R^2 ~ 0.003). The hypothesis is that inversion
acts through the conformational ensemble: it changes which backbone N-H
donors are buried in intramolecular hydrogen bonds, and that is invisible to
a model that only sees the modification pattern.

This script computes 3D descriptors from RDKit conformers of each reference
peptide, adds them to the sequence features, and refits the delta oracle on
D edits alone. If R^2 stays at zero, stereochemistry is not recoverable from
any static representation we have tried.

Usage:
    python conformer_features.py --project . --n-conformers 3
    python conformer_features.py --project . --limit 200   # quick check
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors3D, rdMolDescriptors
from sklearn.ensemble import GradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))

from delta_oracle import DeltaOracle, build_pairs, edit_features, parse_pattern

RDLogger.DisableLog("rdApp.*")

SHAPE_NAMES = (
    "radius_of_gyration",
    "asphericity",
    "eccentricity",
    "inertial_shape_factor",
    "npr1",
    "npr2",
    "spherocity",
    "tpsa",
    "n_imhb",
    "imhb_per_residue",
)


def count_intramolecular_hbonds(mol, conf_id: int) -> int:
    """
    Backbone N-H donors engaged in an intramolecular hydrogen bond.

    Geometric criterion: H to acceptor under 2.5 A with an N-H...A angle
    above 120 degrees. This is the quantity Rezai and Bhardwaj identify as
    controlling passive permeability, and the one an L->D inversion is
    expected to disturb.
    """

    conformer = mol.GetConformer(conf_id)
    positions = conformer.GetPositions()

    donors, acceptors = [], []

    for atom in mol.GetAtoms():

        if atom.GetSymbol() == "N":
            for neighbour in atom.GetNeighbors():
                if neighbour.GetSymbol() == "H":
                    donors.append((atom.GetIdx(), neighbour.GetIdx()))

        if atom.GetSymbol() in ("O", "N"):
            acceptors.append(atom.GetIdx())

    count = 0

    for nitrogen, hydrogen in donors:
        for acceptor in acceptors:

            if acceptor in (nitrogen, hydrogen):
                continue

            if mol.GetBondBetweenAtoms(hydrogen, acceptor) is not None:
                continue

            separation = np.linalg.norm(
                positions[hydrogen] - positions[acceptor]
            )

            if separation > 2.5:
                continue

            a = positions[nitrogen] - positions[hydrogen]
            b = positions[acceptor] - positions[hydrogen]

            cosine = np.dot(a, b) / (
                np.linalg.norm(a) * np.linalg.norm(b) + 1e-9
            )

            if np.degrees(np.arccos(np.clip(cosine, -1, 1))) > 120:
                count += 1
                break

    return count


def shape_features(smiles: str, n_conformers: int, n_residues: int):
    """
    Conformer-ensemble descriptors for one peptide, averaged over conformers.
    """

    molecule = Chem.MolFromSmiles(smiles)

    if molecule is None:
        return None

    molecule = Chem.AddHs(molecule)

    params = AllChem.ETKDGv3()
    params.useMacrocycleTorsions = True
    params.randomSeed = 0xF00D
    params.maxIterations = 200
    params.pruneRmsThresh = 0.5

    ids = AllChem.EmbedMultipleConfs(molecule, n_conformers, params)

    if not len(ids):
        return None

    AllChem.MMFFOptimizeMoleculeConfs(molecule, maxIters=200)

    tpsa = rdMolDescriptors.CalcTPSA(molecule)

    rows = []

    for conf_id in ids:

        hbonds = count_intramolecular_hbonds(molecule, conf_id)

        rows.append(
            [
                Descriptors3D.RadiusOfGyration(molecule, confId=conf_id),
                Descriptors3D.Asphericity(molecule, confId=conf_id),
                Descriptors3D.Eccentricity(molecule, confId=conf_id),
                Descriptors3D.InertialShapeFactor(molecule, confId=conf_id),
                Descriptors3D.NPR1(molecule, confId=conf_id),
                Descriptors3D.NPR2(molecule, confId=conf_id),
                Descriptors3D.SpherocityIndex(molecule, confId=conf_id),
                tpsa,
                float(hbonds),
                hbonds / max(n_residues, 1),
            ]
        )

    return np.asarray(rows).mean(axis=0)


def load_or_build_cache(args, pairs, splits):
    """One descriptor vector per reference peptide, cached to disk."""

    cache_path = Path(args.cache)

    cache = {}

    if cache_path.exists():
        with open(cache_path, "rb") as handle:
            cache = pickle.load(handle)
        print(f"loaded {len(cache):,} cached peptides")

    needed = sorted(set(pairs["reference_id"]))

    if args.limit:
        needed = needed[: args.limit]

    todo = [i for i in needed if i not in cache]

    print(f"{len(needed):,} reference peptides, {len(todo):,} to embed")

    for n, peptide_id in enumerate(todo, 1):

        if peptide_id not in splits.index:
            continue

        pattern = parse_pattern(
            pairs.loc[
                pairs["reference_id"] == peptide_id, "reference_pattern"
            ].iloc[0]
        )

        cache[peptide_id] = shape_features(
            splits.loc[peptide_id, "SMILES"],
            args.n_conformers,
            len(pattern) if pattern else 7,
        )

        if n % 50 == 0:
            print(f"  {n:,}/{len(todo):,}", flush=True)

            with open(cache_path, "wb") as handle:
                pickle.dump(cache, handle)

    with open(cache_path, "wb") as handle:
        pickle.dump(cache, handle)

    return cache


def evaluate(pairs, design, label):
    """Leave-one-parent-out R^2 and top-1 selection."""

    parents = [
        key
        for key, group in pairs.groupby("sidechain_signature")
        if len(group) >= 30
    ]

    scores = []

    for held_out in parents:

        test = (pairs["sidechain_signature"] == held_out).to_numpy()

        if test.sum() < 10 or (~test).sum() < 50:
            continue

        model = GradientBoostingRegressor(random_state=0).fit(
            design[~test], pairs["delta"].to_numpy()[~test]
        )

        predicted = model.predict(design[test])
        truth = pairs["delta"].to_numpy()[test]

        subset = pairs[test].copy()
        subset["predicted"] = predicted

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

        scores.append(
            {
                "r2": 1
                - np.sum((predicted - truth) ** 2)
                / np.sum((truth - truth.mean()) ** 2),
                "top1": hits / groups if groups else np.nan,
                "random": chance / groups if groups else np.nan,
                "n": int(test.sum()),
            }
        )

    frame = pd.DataFrame(scores)

    print(
        f"{label:<24} R2 {frame['r2'].mean():7.4f}   "
        f"top1 {frame['top1'].mean():.3f} (random "
        f"{frame['random'].mean():.3f})   "
        f"{len(frame)} parents, {frame['n'].sum():,} pairs"
    )


def main(args):

    root = Path(args.project)

    splits = pd.concat(
        [
            pd.read_csv(root / "data" / "splits" / f"{name}.csv")
            for name in ("train", "validation", "test")
        ],
        ignore_index=True,
    ).set_index("ID")

    pairs = build_pairs(root)

    cache = load_or_build_cache(args, pairs, splits)

    pairs = pairs[
        pairs["reference_id"].isin(k for k, v in cache.items() if v is not None)
    ].reset_index(drop=True)

    print(f"\n{len(pairs):,} pairs with 3D descriptors\n")

    oracle = DeltaOracle()
    sequence, keep = oracle._design(pairs)

    pairs = pairs.iloc[keep].reset_index(drop=True)

    shape = np.vstack([cache[i] for i in pairs["reference_id"]])
    combined = np.hstack([sequence, shape])

    for edit_type in ("D", "NMe"):

        mask = (pairs["edit_type"] == edit_type).to_numpy()
        subset = pairs[mask].reset_index(drop=True)

        print(f"--- {edit_type} edits ({mask.sum():,} pairs) ---")

        evaluate(subset, sequence[mask], "sequence only")
        evaluate(subset, shape[mask], "3D only")
        evaluate(subset, combined[mask], "sequence + 3D")
        print()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default=".")
    parser.add_argument("--n-conformers", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--cache", default="conformer_cache.pkl")

    main(parser.parse_args())
