"""
T3: can a frozen peptide representation see a backbone edit?

Three evaluations, all on cached embeddings from src/models/embed.py so that
every encoder is compared through an identical head:

1. Global regression on the fixed parent-level split (context only).
2. Leave-one-parent-out ranking on matched single-edit pairs: pairwise sign
   accuracy and within-parent Spearman. LOPO is affordable because the head
   is a ridge regression, and it raises the held-out unit count from 4
   parents to every parent that contributes pairs.
3. Difference-vector coherence, which needs no head at all: if an encoder
   represents an edit as a consistent operation, the embedding differences
   for that edit should point the same way.

Usage:
    python analysis/embedding_study.py --encoders rdkit peptideclm1
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]

SPLIT_DIR = ROOT / "data" / "splits"
EMBED_DIR = ROOT / "results" / "embeddings"
PAIR_PATH = ROOT / "outputs" / "single_edit_pairs.csv"
TABLE_DIR = ROOT / "results" / "tables"

ALPHAS = np.logspace(-2, 4, 25)
RNG = np.random.default_rng(42)


def make_head():
    return make_pipeline(
        StandardScaler(), RidgeCV(alphas=ALPHAS)
    )


# --------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------

def load_peptides() -> pd.DataFrame:

    return pd.concat(
        [pd.read_csv(SPLIT_DIR / f"{n}.csv")
         for n in ["train", "validation", "test"]],
        ignore_index=True,
    )


def load_embeddings(encoder: str):

    data = np.load(EMBED_DIR / f"{encoder}.npz", allow_pickle=True)

    return data["embeddings"], data["ids"]


def first_id(value) -> int:
    return int(str(value).split(",")[0])


def load_pairs(peptides: pd.DataFrame) -> pd.DataFrame:
    """
    Rebuild matched single-edit pairs against the clamped PAMPA values and
    the censoring flags, so deltas are on the same scale the oracle sees.
    """

    pairs = pd.read_csv(PAIR_PATH, low_memory=False)
    pairs = pairs[pairs["clean_single_edit"] == True].copy()

    reference_is_a = pairs["state_a"] == pairs["reference_state"]

    pairs["reference_id"] = np.where(
        reference_is_a,
        pairs["IDs_a"].map(first_id),
        pairs["IDs_b"].map(first_id),
    )

    pairs["modified_id"] = np.where(
        reference_is_a,
        pairs["IDs_b"].map(first_id),
        pairs["IDs_a"].map(first_id),
    )

    lookup = peptides.set_index("ID")

    for side in ["reference", "modified"]:
        ids = pairs[f"{side}_id"]
        pairs[f"{side}_pampa"] = lookup["PAMPA"].reindex(ids).to_numpy()
        pairs[f"{side}_censored"] = (
            lookup["censored"].reindex(ids).to_numpy()
        )

    pairs["split"] = lookup["split"].reindex(pairs["reference_id"]).to_numpy()
    pairs["delta"] = pairs["modified_pampa"] - pairs["reference_pampa"]

    pairs["pair_censored"] = (
        pairs["reference_censored"] | pairs["modified_censored"]
    )

    pairs = pairs[pairs["delta"].notna()].copy()

    # A tie carries no ranking information.
    pairs = pairs[pairs["delta"] != 0].copy()

    return pairs


# --------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------

def regression_metrics(truth, prediction):

    truth = np.asarray(truth, dtype=float)
    prediction = np.asarray(prediction, dtype=float)

    residual = prediction - truth
    denominator = np.sum((truth - truth.mean()) ** 2)

    return {
        "n": int(len(truth)),
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
        "r2": float(1 - np.sum(residual ** 2) / denominator),
        "spearman": float(spearmanr(truth, prediction).statistic),
    }


def global_regression(embeddings, index, peptides):
    """
    Fixed-split regression: fit on train, report on test.
    """

    train = peptides["split"] == "train"
    test = peptides["split"] == "test"

    rows = peptides["ID"].map(index)

    head = make_head()
    head.fit(embeddings[rows[train]], peptides.loc[train, "PAMPA"])

    prediction = head.predict(embeddings[rows[test]])

    overall = regression_metrics(peptides.loc[test, "PAMPA"], prediction)

    clean = ~peptides.loc[test, "censored"].to_numpy()

    uncensored = regression_metrics(
        peptides.loc[test, "PAMPA"].to_numpy()[clean], prediction[clean]
    )

    return {"all": overall, "uncensored": uncensored}


def leave_one_parent_out(embeddings, index, peptides, pairs):
    """
    For each parent contributing pairs: fit the head on every other parent,
    then score that parent's own pairs. The held-out scaffold is never in
    training, so this stays leakage-safe while using all 28 parents.
    """

    parents = sorted(pairs["sidechain_signature"].unique())

    rows = peptides["ID"].map(index)
    labels = peptides["PAMPA"].to_numpy()
    signatures = peptides["sidechain_signature"].to_numpy()

    results = []

    for parent in parents:

        train_mask = signatures != parent

        head = make_head()
        head.fit(embeddings[rows[train_mask]], labels[train_mask])

        subset = pairs[pairs["sidechain_signature"] == parent]

        reference = head.predict(
            embeddings[subset["reference_id"].map(index)]
        )

        modified = head.predict(
            embeddings[subset["modified_id"].map(index)]
        )

        predicted_delta = modified - reference
        true_delta = subset["delta"].to_numpy()

        # A predicted delta of exactly zero is a failure to rank, not a
        # wrong guess: it means the representation cannot see this edit.
        tied = predicted_delta == 0

        correct = np.sign(predicted_delta) == np.sign(true_delta)

        clean = ~subset["pair_censored"].to_numpy()
        decided = ~tied

        # Within-parent Spearman over the measured patterns themselves.
        members = peptides[peptides["sidechain_signature"] == parent]

        if len(members) >= 3:
            member_prediction = head.predict(
                embeddings[members["ID"].map(index)]
            )
            rho = spearmanr(
                members["PAMPA"], member_prediction
            ).statistic
        else:
            rho = np.nan

        results.append(
            {
                "parent": parent,
                "split": subset["split"].iloc[0],
                "source": subset["Source"].iloc[0],
                "n_pairs": int(len(subset)),
                "n_pairs_uncensored": int(clean.sum()),
                "tie_rate": float(tied.mean()),
                "pair_accuracy": float(correct.mean()),
                "pair_accuracy_decided": (
                    float(correct[decided].mean())
                    if decided.any() else np.nan
                ),
                "pair_accuracy_uncensored": (
                    float(correct[clean].mean()) if clean.any() else np.nan
                ),
                "delta_mae": float(
                    np.mean(np.abs(predicted_delta - true_delta))
                ),
                "within_parent_spearman": (
                    float(rho) if rho == rho else np.nan
                ),
                "n_members": int(len(members)),
            }
        )

    return pd.DataFrame(results)


def difference_coherence(embeddings, index, pairs):
    """
    Does one kind of edit move the embedding in one direction?

    For each pair, take the unit difference vector between the modified and
    reference peptide, then measure the mean pairwise cosine similarity
    within an edit type. The null is the same statistic over differences
    between randomly chosen peptides.
    """

    def unit_rows(matrix):
        norm = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.clip(norm, 1e-9, None)

    def mean_pairwise_cosine(vectors, cap=600):

        if len(vectors) < 2:
            return np.nan

        if len(vectors) > cap:
            vectors = vectors[RNG.choice(len(vectors), cap, replace=False)]

        unit = unit_rows(vectors)
        gram = unit @ unit.T

        upper = np.triu_indices(len(unit), k=1)

        return float(gram[upper].mean())

    results = {}

    for edit_type, subset in pairs.groupby("edit_type"):

        difference = (
            embeddings[subset["modified_id"].map(index)]
            - embeddings[subset["reference_id"].map(index)]
        )

        results[edit_type] = {
            "n": int(len(subset)),
            "mean_cosine": mean_pairwise_cosine(difference),
        }

    # Null: differences between random peptide pairs.
    size = min(600, len(embeddings))

    left = RNG.choice(len(embeddings), size, replace=False)
    right = RNG.choice(len(embeddings), size, replace=False)

    results["random_null"] = {
        "n": int(size),
        "mean_cosine": mean_pairwise_cosine(
            embeddings[left] - embeddings[right]
        ),
    }

    return results


# --------------------------------------------------------------------

def run_encoder(encoder, peptides, pairs):

    embeddings, ids = load_embeddings(encoder)

    # Peptides the encoder could not represent come back as NaN rows.
    complete = np.isfinite(embeddings).all(axis=1)

    if not complete.all():
        print(f"  {int((~complete).sum())} peptides had no embedding "
              f"and are excluded")

    ids = ids[complete]
    embeddings = embeddings[complete]

    index = pd.Series(np.arange(len(ids)), index=ids)

    peptides = peptides[peptides["ID"].isin(index.index)].copy()

    usable = pairs[
        pairs["reference_id"].isin(index.index)
        & pairs["modified_id"].isin(index.index)
    ].copy()

    print(f"\n=== {encoder} ===")
    print(f"  embeddings {embeddings.shape[0]:,} x {embeddings.shape[1]:,}")
    print(f"  usable pairs {len(usable):,} of {len(pairs):,}")

    regression = global_regression(embeddings, index, peptides)

    print(f"  fixed-split test  R2 {regression['all']['r2']:7.4f}  "
          f"MAE {regression['all']['mae']:.4f}  "
          f"rho {regression['all']['spearman']:.4f}")

    print(f"  uncensored only   R2 {regression['uncensored']['r2']:7.4f}  "
          f"MAE {regression['uncensored']['mae']:.4f}  "
          f"rho {regression['uncensored']['spearman']:.4f}")

    lopo = leave_one_parent_out(embeddings, index, peptides, usable)

    accuracy = lopo["pair_accuracy"]

    print(f"  LOPO parents {len(lopo)}  "
          f"pair accuracy {accuracy.mean():.4f} +/- {accuracy.std():.4f} "
          f"(median {accuracy.median():.4f})")

    print(f"  tie rate {lopo['tie_rate'].mean():.4f}  "
          f"accuracy among decided pairs "
          f"{lopo['pair_accuracy_decided'].mean():.4f}")

    rho = lopo["within_parent_spearman"].dropna()

    print(f"  within-parent rho {rho.mean():.4f} +/- {rho.std():.4f} "
          f"over {len(rho)} parents")

    coherence = difference_coherence(embeddings, index, usable)

    for key, value in coherence.items():
        print(f"  coherence {key:<12} cos {value['mean_cosine']:7.4f} "
              f"(n={value['n']:,})")

    lopo.insert(0, "encoder", encoder)

    return {
        "encoder": encoder,
        "global_regression": regression,
        "lopo": {
            "n_parents": int(len(lopo)),
            "pair_accuracy_mean": float(accuracy.mean()),
            "pair_accuracy_sd": float(accuracy.std()),
            "pair_accuracy_median": float(accuracy.median()),
            "pair_accuracy_pooled": float(
                np.average(accuracy, weights=lopo["n_pairs"])
            ),
            "tie_rate_mean": float(lopo["tie_rate"].mean()),
            "pair_accuracy_decided_mean": float(
                lopo["pair_accuracy_decided"].mean()
            ),
            "within_parent_spearman_mean": float(rho.mean()),
            "within_parent_spearman_sd": float(rho.std()),
        },
        "difference_coherence": coherence,
    }, lopo


def main(args):

    peptides = load_peptides()
    pairs = load_pairs(peptides)

    print(f"peptides {len(peptides):,} | "
          f"clean single-edit pairs {len(pairs):,} | "
          f"parents with pairs {pairs['sidechain_signature'].nunique()}")

    print(pairs.groupby("split")["sidechain_signature"]
          .agg(["count", "nunique"]).to_string())

    summaries = []
    tables = []

    for encoder in args.encoders:

        summary, lopo = run_encoder(encoder, peptides, pairs)

        summaries.append(summary)
        tables.append(lopo)

    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    pd.concat(tables, ignore_index=True).to_csv(
        TABLE_DIR / "embedding_study_per_parent.csv", index=False
    )

    with open(TABLE_DIR / "embedding_study.json", "w") as handle:
        json.dump(summaries, handle, indent=2)

    overview = pd.DataFrame(
        [
            {
                "encoder": s["encoder"],
                "test_r2": s["global_regression"]["all"]["r2"],
                "test_mae": s["global_regression"]["all"]["mae"],
                "test_rho": s["global_regression"]["all"]["spearman"],
                "pair_acc_mean": s["lopo"]["pair_accuracy_mean"],
                "pair_acc_sd": s["lopo"]["pair_accuracy_sd"],
                "pair_acc_pooled": s["lopo"]["pair_accuracy_pooled"],
                "tie_rate": s["lopo"]["tie_rate_mean"],
                "pair_acc_decided": s["lopo"][
                    "pair_accuracy_decided_mean"
                ],
                "within_parent_rho": s["lopo"][
                    "within_parent_spearman_mean"
                ],
                "coh_NMe": s["difference_coherence"]
                .get("NMe", {})
                .get("mean_cosine"),
                "coh_D": s["difference_coherence"]
                .get("D", {})
                .get("mean_cosine"),
                "coh_null": s["difference_coherence"]["random_null"][
                    "mean_cosine"
                ],
            }
            for s in summaries
        ]
    ).round(4)

    overview.to_csv(TABLE_DIR / "embedding_study_summary.csv", index=False)

    print("\n=== summary ===")
    print(overview.to_string(index=False))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--encoders", nargs="+", default=["rdkit"])

    main(parser.parse_args())
