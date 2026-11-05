"""
T3 step 1: extract a frozen embedding matrix for every peptide in the splits.

Each encoder is run once, with no fine-tuning, and the resulting matrix is
cached to results/embeddings/<name>.npz alongside the peptide IDs. The
evaluation in analysis/embedding_study.py consumes these caches, so the
expensive GPU pass happens once per encoder.

Usage:
    python src/models/embed.py --encoder rdkit          # CPU
    python src/models/embed.py --encoder peptideclm1    # GPU
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

SPLIT_DIR = PROJECT_ROOT / "data" / "splits"
EMBED_DIR = PROJECT_ROOT / "results" / "embeddings"

TRANSFORMERS = {
    "peptideclm1": "aaronfeller/PeptideCLM-23M-all",
    "peptideclm2-mtr": "aaronfeller/peptideclm-2-mtr-small",
    "peptideclm2-hybrid": "aaronfeller/peptideclm-2-hybrid-small",
    "helmbert": "Flansma/helm-bert",
}


def load_peptides() -> pd.DataFrame:

    frames = []

    for name in ["train", "validation", "test"]:
        frame = pd.read_csv(SPLIT_DIR / f"{name}.csv")
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------
# RDKit descriptors
# --------------------------------------------------------------------

def embed_rdkit(peptides: pd.DataFrame):
    """
    2D descriptors plus a Morgan fingerprint. The literature reports these
    as competitive with chemical language models, so this is a real
    baseline rather than a formality.
    """

    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors
    from rdkit.Chem import rdFingerprintGenerator

    RDLogger.DisableLog("rdApp.*")

    # Ipc grows exponentially with molecule size and reaches ~1e25 here,
    # which swamps every other feature after scaling.
    unstable = {"Ipc"}

    names = [
        name for name, _ in Descriptors.descList if name not in unstable
    ]

    calculator = Descriptors.CalcMolDescriptors

    # Chirality must be on: without it a peptide and its D-epimer get
    # byte-identical fingerprints, and every L->D edit becomes invisible.
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=1024, includeChirality=True
    )

    vectors = []
    failed = []

    for row in peptides.itertuples(index=False):

        molecule = Chem.MolFromSmiles(row.SMILES)

        if molecule is None:
            failed.append(row.ID)
            vectors.append(None)
            continue

        values = calculator(molecule)
        descriptors = np.array(
            [values.get(name, np.nan) for name in names], dtype=np.float64
        )

        fingerprint = np.asarray(
            generator.GetFingerprintAsNumPy(molecule), dtype=np.float64
        )

        vectors.append(np.concatenate([descriptors, fingerprint]))

    width = next(v.shape[0] for v in vectors if v is not None)

    matrix = np.vstack(
        [np.full(width, np.nan) if v is None else v for v in vectors]
    )

    # Descriptors can emit inf or nan on exotic structures.
    matrix[~np.isfinite(matrix)] = np.nan

    column_median = np.nanmedian(matrix, axis=0)
    column_median[~np.isfinite(column_median)] = 0.0

    missing = ~np.isfinite(matrix)
    matrix[missing] = np.take(column_median, np.where(missing)[1])

    feature_names = names + [f"morgan_{i}" for i in range(1024)]

    return matrix, feature_names, failed


# --------------------------------------------------------------------
# Transformer encoders
# --------------------------------------------------------------------

def embed_transformer(peptides: pd.DataFrame, encoder: str, batch_size: int):

    import torch
    from transformers import AutoModel, AutoTokenizer

    model_id = TRANSFORMERS[encoder]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if encoder == "peptideclm1":
        from external.PeptideCLM.tokenizer.my_tokenizers import (
            SMILES_SPE_Tokenizer,
        )

        tokenizer_dir = PROJECT_ROOT / "external" / "PeptideCLM" / "tokenizer"

        tokenizer = SMILES_SPE_Tokenizer(
            str(tokenizer_dir / "new_vocab.txt"),
            str(tokenizer_dir / "new_splits.txt"),
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True
        )

    model = AutoModel.from_pretrained(
        model_id, trust_remote_code=True
    ).to(device)

    model.eval()

    # HELM-BERT reads HELM notation; the others read SMILES.
    column = "HELM" if encoder == "helmbert" else "SMILES"

    def tokenize(text):

        if encoder == "peptideclm1":
            tokens = tokenizer.tokenize(text)

            fixed = []
            for token in tokens:
                fixed.extend(["ccc", "c"] if token == "cccc" else [token])

            ids = tokenizer.convert_tokens_to_ids(fixed)

            if tokenizer.unk_token_id in ids:
                return None

            return tokenizer.build_inputs_with_special_tokens(ids)

        return tokenizer(
            text, truncation=False, padding=False
        )["input_ids"]

    encoded = [tokenize(text) for text in peptides[column]]

    failed = [
        row.ID
        for row, ids in zip(peptides.itertuples(index=False), encoded)
        if ids is None
    ]

    usable = [i for i, ids in enumerate(encoded) if ids is not None]

    # Unknown tokens are recorded, not treated as failures: HELM-BERT maps
    # only truly unseen characters to [UNK], so dropping those peptides
    # would shrink the comparison set for one encoder and not the others.
    unk_id = tokenizer.unk_token_id

    has_unk = np.array(
        [
            ids is not None and unk_id in ids
            for ids in encoded
        ]
    )

    if has_unk.any():
        print(f"  {int(has_unk.sum())} sequences contain [UNK] (kept)")

    longest = max(len(encoded[i]) for i in usable)

    limit = getattr(model.config, "max_position_embeddings", None)

    if limit and longest > limit:
        print(f"  WARNING: longest sequence {longest} exceeds the "
              f"model limit {limit}; those peptides will be truncated")

        usable = [i for i in usable if len(encoded[i]) <= limit]

    width = model.config.hidden_size
    matrix = np.full((len(peptides), width), np.nan)

    pad_id = tokenizer.pad_token_id

    with torch.no_grad():

        for start in range(0, len(usable), batch_size):

            indices = usable[start:start + batch_size]
            sequences = [encoded[i] for i in indices]

            longest = max(len(s) for s in sequences)

            ids = torch.tensor(
                [s + [pad_id] * (longest - len(s)) for s in sequences],
                dtype=torch.long,
            ).to(device)

            mask = torch.tensor(
                [[1] * len(s) + [0] * (longest - len(s)) for s in sequences],
                dtype=torch.long,
            ).to(device)

            hidden = model(
                input_ids=ids, attention_mask=mask
            ).last_hidden_state

            expanded = mask.unsqueeze(-1).float()

            pooled = (
                (hidden * expanded).sum(1)
                / expanded.sum(1).clamp(min=1e-9)
            )

            matrix[indices] = pooled.cpu().numpy()

            print(
                f"  embedded {min(start + batch_size, len(usable)):5,}"
                f" / {len(usable):,}",
                end="\r",
                flush=True,
            )

    print()

    names = [f"dim_{i}" for i in range(width)]

    return matrix, names, failed, has_unk


# --------------------------------------------------------------------

def main(args):

    peptides = load_peptides()

    print(f"=== embedding {len(peptides):,} peptides with "
          f"{args.encoder} ===")

    if args.encoder == "rdkit":
        matrix, names, failed = embed_rdkit(peptides)
        has_unk = np.zeros(len(peptides), dtype=bool)
    else:
        matrix, names, failed, has_unk = embed_transformer(
            peptides, args.encoder, args.batch_size
        )

    print(f"matrix: {matrix.shape[0]:,} x {matrix.shape[1]:,}")
    print(f"failed to encode: {len(failed)}")

    EMBED_DIR.mkdir(parents=True, exist_ok=True)

    path = EMBED_DIR / f"{args.encoder}.npz"

    np.savez_compressed(
        path,
        embeddings=matrix,
        ids=peptides["ID"].to_numpy(),
        feature_names=np.array(names, dtype=object),
        failed=np.array(failed, dtype=object),
        has_unk=has_unk,
    )

    print(f"wrote {path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--encoder",
        choices=["rdkit"] + sorted(TRANSFORMERS),
        required=True,
    )
    parser.add_argument("--batch-size", type=int, default=32)

    main(parser.parse_args())
