
import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification

PROJECT_ROOT = Path(__file__).resolve().parents[2]

import sys
sys.path.insert(0, str(PROJECT_ROOT))

from external.PeptideCLM.tokenizer.my_tokenizers import SMILES_SPE_Tokenizer


def encode_smiles(tokenizer, smiles):
    tokens = tokenizer.tokenize(smiles)

    fixed = []
    for token in tokens:
        fixed.extend(["ccc", "c"] if token == "cccc" else [token])

    ids = tokenizer.convert_tokens_to_ids(fixed)

    if tokenizer.unk_token_id in ids:
        return None

    return tokenizer.build_inputs_with_special_tokens(ids)


class PeptideDataset(Dataset):
    def __init__(self, dataframe, tokenizer):
        self.records = []
        excluded = 0

        for row in dataframe.itertuples(index=False):
            ids = encode_smiles(tokenizer, row.SMILES)

            if ids is None:
                excluded += 1
                continue

            self.records.append({
                "input_ids": ids,
                "smiles": row.SMILES,
                "label": float(row.PAMPA),
            })

        print(f"Retained {len(self.records)} | Excluded {excluded}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]


def make_collator(pad_id):
    def collate(batch):
        max_len = max(len(x["input_ids"]) for x in batch)

        ids = []
        masks = []

        for x in batch:
            sequence = x["input_ids"]
            padding = max_len - len(sequence)

            ids.append(sequence + [pad_id] * padding)
            masks.append([1] * len(sequence) + [0] * padding)

        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
            "labels": torch.tensor(
                [x["label"] for x in batch], dtype=torch.float32
            ),
            "smiles": [x["smiles"] for x in batch],
        }

    return collate


def evaluate(model, loader, device):
    model.eval()

    predictions = []
    labels = []
    smiles = []

    with torch.no_grad():
        for batch in loader:
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )

            predictions.extend(outputs.logits.squeeze(-1).cpu().tolist())
            labels.extend(batch["labels"].tolist())
            smiles.extend(batch["smiles"])

    predictions = np.asarray(predictions)
    labels = np.asarray(labels)

    mae = float(np.mean(np.abs(predictions - labels)))
    rmse = float(np.sqrt(np.mean((predictions - labels) ** 2)))

    denominator = np.sum((labels - labels.mean()) ** 2)
    r2 = float(
        1 - np.sum((labels - predictions) ** 2) / denominator
    ) if denominator > 0 else float("nan")

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    }, pd.DataFrame({
        "SMILES": smiles,
        "PAMPA": labels,
        "prediction": predictions,
    })


def main(args):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("GPU required. Run through a GPU Slurm allocation.")

    print("Device:", device)
    print("GPU:", torch.cuda.get_device_name(0))

    split_dir = PROJECT_ROOT / "data/processed/peptideclm_v1"
    output_dir = PROJECT_ROOT / "results/peptideclm_v1"
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_dir = PROJECT_ROOT / "external/PeptideCLM/tokenizer"
    tokenizer = SMILES_SPE_Tokenizer(
        str(tokenizer_dir / "new_vocab.txt"),
        str(tokenizer_dir / "new_splits.txt"),
    )

    datasets = {}
    loaders = {}

    for split in ["train", "val", "test"]:
        df = pd.read_csv(split_dir / f"{split}.csv")
        datasets[split] = PeptideDataset(df, tokenizer)

        loaders[split] = DataLoader(
            datasets[split],
            batch_size=args.batch_size,
            shuffle=(split == "train"),
            collate_fn=make_collator(tokenizer.pad_token_id),
        )

    model = AutoModelForSequenceClassification.from_pretrained(
        "aaronfeller/PeptideCLM-23M-all",
        num_labels=1,
        problem_type="regression",
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    best_mae = float("inf")
    best_path = output_dir / "best_model"

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)

            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
            )

            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        val_metrics, _ = evaluate(model, loaders["val"], device)

        print(
            f"Epoch {epoch} | "
            f"Train MSE: {total_loss / len(loaders['train']):.4f} | "
            f"Val MAE: {val_metrics['mae']:.4f} | "
            f"Val RMSE: {val_metrics['rmse']:.4f} | "
            f"Val R2: {val_metrics['r2']:.4f}",
            flush=True,
        )

        if val_metrics["mae"] < best_mae:
            best_mae = val_metrics["mae"]
            model.save_pretrained(best_path)
            print("Saved best checkpoint.", flush=True)

    # Evaluate the untouched test cluster once, using the best checkpoint.
    model = AutoModelForSequenceClassification.from_pretrained(
        best_path
    ).to(device)

    test_metrics, test_predictions = evaluate(
        model, loaders["test"], device
    )

    print("\nFinal test metrics:", test_metrics)

    test_predictions.to_csv(
        output_dir / "test_predictions.csv", index=False
    )

    with open(output_dir / "metrics.json", "w") as f:
        json.dump({
            "validation_best_mae": best_mae,
            "test": test_metrics,
            "training": vars(args),
            "seed": 42,
            "split": {
                "train_clusters": [3, 4, 5, 6],
                "validation_cluster": 2,
                "test_cluster": 1,
            },
        }, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)

    main(parser.parse_args())

