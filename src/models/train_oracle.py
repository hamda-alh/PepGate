"""
T2: PAMPA oracle trained on the parent-level split from data/splits/.

Supports both PeptideCLM generations. Each keeps the regression head it used
in the original pilot, so the only controlled change relative to those runs is
the split and the censoring treatment.

Usage:
    python src/models/train_oracle.py --model peptideclm1
    python src/models/train_oracle.py --model peptideclm2-mtr
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from torch import nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

SPLIT_DIR = PROJECT_ROOT / "data" / "splits"

MODELS = {
    "peptideclm1": {
        "hf_id": "aaronfeller/PeptideCLM-23M-all",
        "family": "clm1",
    },
    "peptideclm2-mtr": {
        "hf_id": "aaronfeller/peptideclm-2-mtr-small",
        "family": "clm2",
    },
    "peptideclm2-hybrid": {
        "hf_id": "aaronfeller/peptideclm-2-hybrid-small",
        "family": "clm2",
    },
}


# --------------------------------------------------------------------
# Tokenization
# --------------------------------------------------------------------

def build_tokenizer(family, hf_id):

    if family == "clm1":
        from external.PeptideCLM.tokenizer.my_tokenizers import (
            SMILES_SPE_Tokenizer,
        )

        tokenizer_dir = PROJECT_ROOT / "external" / "PeptideCLM" / "tokenizer"

        return SMILES_SPE_Tokenizer(
            str(tokenizer_dir / "new_vocab.txt"),
            str(tokenizer_dir / "new_splits.txt"),
        )

    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)


def encode(tokenizer, family, smiles):
    """
    Return token ids, or None when the SMILES cannot be represented.
    """

    if family == "clm1":
        tokens = tokenizer.tokenize(smiles)

        fixed = []
        for token in tokens:
            fixed.extend(["ccc", "c"] if token == "cccc" else [token])

        ids = tokenizer.convert_tokens_to_ids(fixed)

        if tokenizer.unk_token_id in ids:
            return None

        return tokenizer.build_inputs_with_special_tokens(ids)

    ids = tokenizer(smiles, truncation=False, padding=False)["input_ids"]

    if tokenizer.unk_token_id in ids:
        return None

    return ids


# --------------------------------------------------------------------
# Data
# --------------------------------------------------------------------

class PeptideDataset(Dataset):

    def __init__(self, frame, tokenizer, family, name):

        self.records = []
        dropped = []

        for row in frame.itertuples(index=False):

            ids = encode(tokenizer, family, row.SMILES)

            if ids is None:
                dropped.append(row.ID)
                continue

            self.records.append(
                {
                    "input_ids": ids,
                    "id": row.ID,
                    "label": float(row.PAMPA),
                    "censored": bool(row.censored),
                    "parent": row.sidechain_signature,
                }
            )

        self.dropped = dropped

        print(f"  {name:<11} kept {len(self.records):5,}  "
              f"dropped {len(dropped):3,} (untokenizable)")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]


def make_collator(pad_id):

    def collate(batch):

        max_len = max(len(x["input_ids"]) for x in batch)

        ids, masks = [], []

        for item in batch:
            sequence = item["input_ids"]
            padding = max_len - len(sequence)

            ids.append(sequence + [pad_id] * padding)
            masks.append([1] * len(sequence) + [0] * padding)

        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
            "labels": torch.tensor(
                [x["label"] for x in batch], dtype=torch.float32
            ),
            "ids": [x["id"] for x in batch],
            "censored": [x["censored"] for x in batch],
            "parents": [x["parent"] for x in batch],
        }

    return collate


# --------------------------------------------------------------------
# Model
# --------------------------------------------------------------------

class CLM2Regressor(nn.Module):
    """
    Matches the head used in the original PeptideCLM-2 pilot.
    """

    def __init__(self, model_id, dropout=0.1):
        super().__init__()

        from transformers import AutoModel

        self.encoder = AutoModel.from_pretrained(
            model_id, trust_remote_code=True
        )

        hidden = self.encoder.config.hidden_size

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, input_ids, attention_mask):

        output = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        )

        hidden = output.last_hidden_state

        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

        return self.head(pooled).squeeze(-1)


def build_model(family, hf_id, device):

    if family == "clm1":
        from transformers import AutoModelForSequenceClassification

        model = AutoModelForSequenceClassification.from_pretrained(
            hf_id, num_labels=1, problem_type="regression"
        )

        return model.to(device)

    return CLM2Regressor(hf_id).to(device)


def predict(model, family, batch, device):

    if family == "clm1":
        output = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
        )

        return output.logits.squeeze(-1)

    return model(
        input_ids=batch["input_ids"].to(device),
        attention_mask=batch["attention_mask"].to(device),
    )


# --------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------

def regression_metrics(labels, predictions):

    labels = np.asarray(labels, dtype=float)
    predictions = np.asarray(predictions, dtype=float)

    if len(labels) < 2:
        return {"n": int(len(labels))}

    residual = predictions - labels
    denominator = np.sum((labels - labels.mean()) ** 2)

    rho = spearmanr(labels, predictions).statistic

    return {
        "n": int(len(labels)),
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
        "r2": (
            float(1 - np.sum(residual ** 2) / denominator)
            if denominator > 0 else float("nan")
        ),
        "spearman": float(rho) if rho == rho else float("nan"),
        "bias": float(np.mean(residual)),
    }


def evaluate(model, family, loader, device):

    model.eval()

    rows = []

    with torch.no_grad():
        for batch in loader:

            values = predict(model, family, batch, device)

            rows.extend(
                {
                    "ID": i,
                    "PAMPA": label,
                    "prediction": value,
                    "censored": censored,
                    "sidechain_signature": parent,
                }
                for i, label, value, censored, parent in zip(
                    batch["ids"],
                    batch["labels"].tolist(),
                    values.cpu().tolist(),
                    batch["censored"],
                    batch["parents"],
                )
            )

    frame = pd.DataFrame(rows)

    overall = regression_metrics(frame["PAMPA"], frame["prediction"])

    uncensored = frame[~frame["censored"]]

    return {
        "all": overall,
        "uncensored": regression_metrics(
            uncensored["PAMPA"], uncensored["prediction"]
        ),
    }, frame


# --------------------------------------------------------------------
# Training
# --------------------------------------------------------------------

def run_seed(args, config, splits, device, seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    family = config["family"]

    tokenizer = build_tokenizer(family, config["hf_id"])
    collate = make_collator(tokenizer.pad_token_id)

    print(f"\n--- seed {seed} ---")

    loaders = {}

    for name in ["train", "validation", "test"]:

        frame = splits[name]

        if name == "train" and args.exclude_censored_train:
            frame = frame[~frame["censored"]]

        dataset = PeptideDataset(frame, tokenizer, family, name)

        loaders[name] = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=(name == "train"),
            collate_fn=collate,
        )

    model = build_model(family, config["hf_id"], device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    best_mae = float("inf")
    best_epoch = 0
    best_state = None
    stale = 0

    for epoch in range(1, args.epochs + 1):

        model.train()
        total = 0.0

        for batch in loaders["train"]:

            optimizer.zero_grad(set_to_none=True)

            values = predict(model, family, batch, device)
            loss = loss_fn(values, batch["labels"].to(device))

            loss.backward()
            optimizer.step()

            total += loss.item()

        metrics, _ = evaluate(model, family, loaders["validation"], device)
        val = metrics["all"]

        print(
            f"  epoch {epoch:2d} | train MSE {total / len(loaders['train']):7.4f} "
            f"| val MAE {val['mae']:.4f} | val R2 {val['r2']:7.4f} "
            f"| val rho {val['spearman']:.4f}",
            flush=True,
        )

        if val["mae"] < best_mae - 1e-4:
            best_mae = val["mae"]
            best_epoch = epoch
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1

            if stale >= args.patience:
                print(f"  early stop (no gain for {args.patience} epochs)")
                break

    model.load_state_dict(best_state)

    test_metrics, test_predictions = evaluate(
        model, family, loaders["test"], device
    )

    test_predictions["seed"] = seed

    print(
        f"  best epoch {best_epoch} | test MAE "
        f"{test_metrics['all']['mae']:.4f} | test R2 "
        f"{test_metrics['all']['r2']:.4f} | test rho "
        f"{test_metrics['all']['spearman']:.4f}"
    )

    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "validation_best_mae": best_mae,
        "test": test_metrics,
    }, test_predictions


def constant_baseline(splits):
    """
    Predict the training mean for every test peptide.

    The constant comes from training labels, never from the test set.
    """

    constant = splits["train"]["PAMPA"].mean()

    test = splits["test"]

    predictions = np.full(len(test), constant)

    overall = regression_metrics(test["PAMPA"], predictions)

    uncensored = test[~test["censored"]]

    return {
        "train_mean": float(constant),
        "all": overall,
        "uncensored": regression_metrics(
            uncensored["PAMPA"],
            np.full(len(uncensored), constant),
        ),
    }


def main(args):

    config = MODELS[args.model]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type != "cuda":
        raise RuntimeError("GPU required. Submit through a GPU allocation.")

    print(f"=== T2 oracle: {args.model} ({config['hf_id']}) ===")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    splits = {
        name: pd.read_csv(SPLIT_DIR / f"{name}.csv")
        for name in ["train", "validation", "test"]
    }

    for name, frame in splits.items():
        print(f"  {name:<11} {len(frame):5,} peptides  "
              f"{frame['censored'].sum():3,} censored  "
              f"{frame['sidechain_signature'].nunique():4,} parents")

    baseline = constant_baseline(splits)

    print(f"\nConstant baseline (train mean = {baseline['train_mean']:.4f})")
    print(f"  all        MAE {baseline['all']['mae']:.4f}  "
          f"RMSE {baseline['all']['rmse']:.4f}")
    print(f"  uncensored MAE {baseline['uncensored']['mae']:.4f}  "
          f"RMSE {baseline['uncensored']['rmse']:.4f}")

    output_dir = PROJECT_ROOT / "results" / f"oracle_{args.model}"
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = []
    predictions = []

    for seed in args.seeds:
        summary, frame = run_seed(args, config, splits, device, seed)
        runs.append(summary)
        predictions.append(frame)

    all_predictions = pd.concat(predictions, ignore_index=True)
    all_predictions.to_csv(output_dir / "test_predictions.csv", index=False)

    print("\n=== across seeds ===")

    for scope in ["all", "uncensored"]:
        for metric in ["mae", "rmse", "r2", "spearman", "bias"]:

            values = [r["test"][scope][metric] for r in runs]

            print(f"  {scope:<11} {metric:<9} "
                  f"{np.mean(values):8.4f} +/- {np.std(values):.4f}")

    with open(output_dir / "metrics.json", "w") as handle:
        json.dump(
            {
                "model": config["hf_id"],
                "split": "parent_level (data/splits)",
                "censoring": "clamped at -8.5, flagged",
                "exclude_censored_train": args.exclude_censored_train,
                "training": {
                    "epochs_max": args.epochs,
                    "patience": args.patience,
                    "batch_size": args.batch_size,
                    "lr": args.lr,
                    "seeds": args.seeds,
                },
                "constant_baseline": baseline,
                "runs": runs,
            },
            handle,
            indent=2,
        )

    print(f"\nWrote {output_dir}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model", choices=sorted(MODELS), default="peptideclm1"
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[42, 43, 44]
    )
    parser.add_argument(
        "--exclude-censored-train", action="store_true"
    )

    main(parser.parse_args())
