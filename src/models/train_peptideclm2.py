
import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_ID = "aaronfeller/peptideclm-2-hybrid-small"


class PeptideDataset(Dataset):
    def __init__(self, csv_path, tokenizer):
        df = pd.read_csv(csv_path)

        self.smiles = df["SMILES"].tolist()
        self.labels = df["PAMPA"].astype(float).tolist()

        self.encodings = tokenizer(
            self.smiles,
            truncation=False,
            padding=False
        )["input_ids"]

        for ids in self.encodings:
            if tokenizer.unk_token_id in ids:
                raise ValueError("Dataset contains an unknown token.")

        print(f"{csv_path.name}: {len(self.smiles)} records")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings[idx],
            "label": self.labels[idx],
            "smiles": self.smiles[idx]
        }


def make_collator(pad_id):
    def collate(batch):
        max_len = max(len(x["input_ids"]) for x in batch)

        ids, masks = [], []

        for item in batch:
            seq = item["input_ids"]
            pad = max_len - len(seq)

            ids.append(seq + [pad_id] * pad)
            masks.append([1] * len(seq) + [0] * pad)

        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
            "labels": torch.tensor(
                [x["label"] for x in batch], dtype=torch.float32
            ),
            "smiles": [x["smiles"] for x in batch]
        }

    return collate


class PeptideCLM2Regressor(nn.Module):
    def __init__(self, model_id, dropout=0.1):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True
        )

        hidden_dim = 512

        self.regression_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        pooled = outputs.mean_pool

        return self.regression_head(pooled).squeeze(-1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    predictions, labels, smiles = [], [], []

    for batch in loader:
        preds = model(
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device)
        )

        predictions.extend(preds.cpu().tolist())
        labels.extend(batch["labels"].tolist())
        smiles.extend(batch["smiles"])

    y = np.asarray(labels)
    yhat = np.asarray(predictions)

    mae = float(np.mean(np.abs(y - yhat)))
    rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))

    denominator = np.sum((y - y.mean()) ** 2)
    r2 = float(
        1 - np.sum((y - yhat) ** 2) / denominator
    ) if denominator > 0 else float("nan")

    metrics = {
        "mae": mae,
        "rmse": rmse,
        "r2": r2
    }

    results = pd.DataFrame({
        "SMILES": smiles,
        "PAMPA": y,
        "prediction": yhat
    })

    return metrics, results


def main(args):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required.")

    device = torch.device("cuda")

    data_dir = PROJECT_ROOT / "data/processed/peptideclm2_v1"
    output_dir = PROJECT_ROOT / "results/peptideclm2_v1"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    print("Model:", MODEL_ID, flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        trust_remote_code=True
    )

    collator = make_collator(tokenizer.pad_token_id)

    loaders = {}

    for name in ["train", "val", "test"]:
        dataset = PeptideDataset(
            data_dir / f"{name}.csv",
            tokenizer
        )

        loaders[name] = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=(name == "train"),
            collate_fn=collator,
            num_workers=0
        )

    model = PeptideCLM2Regressor(MODEL_ID).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr
    )

    criterion = nn.MSELoss()

    best_mae = float("inf")
    best_epoch = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_samples = 0

        for batch in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)

            predictions = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device)
            )

            labels = batch["labels"].to(device)
            loss = criterion(predictions, labels)

            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

        train_mse = total_loss / total_samples

        val_metrics, _ = evaluate(
            model, loaders["val"], device
        )

        print(
            f"Epoch {epoch} | "
            f"Train MSE: {train_mse:.4f} | "
            f"Val MAE: {val_metrics['mae']:.4f} | "
            f"Val RMSE: {val_metrics['rmse']:.4f} | "
            f"Val R2: {val_metrics['r2']:.4f}",
            flush=True
        )

        if val_metrics["mae"] < best_mae:
            best_mae = val_metrics["mae"]
            best_epoch = epoch

            torch.save(
                model.state_dict(),
                output_dir / "best_model.pt"
            )

            print("Saved best checkpoint.", flush=True)

    # Reload the validation-selected checkpoint
    model.load_state_dict(
        torch.load(
            output_dir / "best_model.pt",
            map_location=device,
            weights_only=True
        )
    )

    # Evaluate the held-out test set once
    test_metrics, test_predictions = evaluate(
        model, loaders["test"], device
    )

    test_predictions.to_csv(
        output_dir / "test_predictions.csv",
        index=False
    )

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(
            {
                "model": MODEL_ID,
                "seed": 42,
                "best_epoch": best_epoch,
                "best_validation_mae": best_mae,
                "test": test_metrics,
                "training": vars(args),
                "split": {
                    "train_clusters": [3, 4, 5, 6],
                    "validation_cluster": 2,
                    "test_cluster": 1
                }
            },
            f,
            indent=2
        )

    print("Final test metrics:", test_metrics, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)

    main(parser.parse_args())

