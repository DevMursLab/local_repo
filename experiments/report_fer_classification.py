"""Produce publication-ready metrics from real held-out FER predictions.

The input CSV must contain y_true and y_pred. This script never generates
predictions and should be run only on a frozen held-out prediction export.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


def bootstrap_accuracy_ci(y_true, y_pred, draws, seed):
    rng = np.random.default_rng(seed)
    indices = np.arange(len(y_true))
    # Vectorised resampling avoids repeatedly constructing sklearn objects for
    # thousands of bootstrap draws.
    sampled = rng.choice(indices, (draws, len(indices)), replace=True)
    scores = (y_true[sampled] == y_pred[sampled]).mean(axis=1)
    return [float(value) for value in np.percentile(scores, [2.5, 97.5])]


def main():
    parser = argparse.ArgumentParser(description="Summarise real FER test predictions.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261107)
    args = parser.parse_args()

    with Path(args.predictions).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not {"y_true", "y_pred"}.issubset(rows[0]):
        raise ValueError("Predictions CSV must have non-empty y_true and y_pred columns.")

    labels = sorted({row["y_true"] for row in rows} | {row["y_pred"] for row in rows})
    y_true = np.array([row["y_true"] for row in rows])
    y_pred = np.array([row["y_pred"] for row in rows])
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    report = {
        "n_test": len(rows),
        "labels": labels,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "accuracy_bootstrap_95_ci": bootstrap_accuracy_ci(y_true, y_pred, args.bootstrap_draws, args.seed),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "per_class": [
            {"label": label, "precision": float(p), "recall": float(r), "f1": float(f), "support": int(s)}
            for label, p, r, f, s in zip(labels, precision, recall, f1, support)
        ],
        "bootstrap_draws": args.bootstrap_draws,
        "bootstrap_seed": args.seed,
    }
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    (output / "classification_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (output / "per_class_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["label", "precision", "recall", "f1", "support"])
        writer.writeheader(); writer.writerows(report["per_class"])
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
