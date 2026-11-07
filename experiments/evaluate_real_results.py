"""Reproducible evaluation from held-out, real experimental predictions.

This program intentionally never simulates labels, predictions, interactions, or
audio features. It is the only supported route for producing paper numbers.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import accuracy_score, cohen_kappa_score, precision_recall_fscore_support


def read_classification(path: Path):
    required = {"sample_id", "y_true", "y_pred"}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"{path} must contain: {', '.join(sorted(required))}")
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError("sample_id values must be unique; evaluate a held-out split once.")
    return rows


def classification_metrics(rows):
    y_true = [row["y_true"] for row in rows]
    y_pred = [row["y_pred"] for row in rows]
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {"n_held_out_samples": len(rows), "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_precision": float(precision), "macro_recall": float(recall), "macro_f1": float(f1),
            "cohen_kappa": float(cohen_kappa_score(y_true, y_pred))}


def bootstrap_accuracy_ci(rows, seed, repetitions=2000):
    rng = np.random.default_rng(seed)
    correct = np.asarray([row["y_true"] == row["y_pred"] for row in rows], dtype=float)
    values = np.asarray([rng.choice(correct, size=len(correct), replace=True).mean() for _ in range(repetitions)])
    return [float(x) for x in np.quantile(values, [0.025, 0.975])]


def read_recommendations(path: Path):
    required = {"session_id", "method", "ndcg_at_5"}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"{path} must contain: {', '.join(sorted(required))}")
    results = defaultdict(dict)
    for row in rows:
        if row["method"] in results[row["session_id"]]:
            raise ValueError("Each session/method pair must occur only once.")
        results[row["session_id"]][row["method"]] = float(row["ndcg_at_5"])
    return results


def paired_recommendation_test(sessions, proposed, baseline):
    paired = [(x[proposed], x[baseline]) for x in sessions.values() if proposed in x and baseline in x]
    if len(paired) < 10:
        raise ValueError("At least 10 paired real sessions are required.")
    ours, base = np.asarray(paired).T
    delta = ours - base
    p_value = 1.0 if np.allclose(delta, 0) else stats.wilcoxon(ours, base, alternative="two-sided").pvalue
    return {"n_paired_sessions": len(paired), "proposed_mean_ndcg_at_5": float(ours.mean()),
            "baseline_mean_ndcg_at_5": float(base.mean()), "mean_paired_difference": float(delta.mean()),
            "wilcoxon_p_value_two_sided": float(p_value)}


def main():
    parser = argparse.ArgumentParser(description="Evaluate held-out AffectiveEcho results without synthetic data.")
    parser.add_argument("--classification", type=Path, required=True)
    parser.add_argument("--recommendations", type=Path, required=True)
    parser.add_argument("--proposed-method", required=True)
    parser.add_argument("--baseline-method", required=True)
    parser.add_argument("--output", type=Path, default=Path("experiments/real_evaluation_results.json"))
    parser.add_argument("--seed", type=int, default=20261107)
    args = parser.parse_args()
    rows = read_classification(args.classification)
    output = {"provenance": {"synthetic_data_used": False, "classification_input": str(args.classification),
               "recommendation_input": str(args.recommendations), "bootstrap_seed": args.seed},
              "classification": classification_metrics(rows),
              "recommendation": paired_recommendation_test(read_recommendations(args.recommendations), args.proposed_method, args.baseline_method)}
    output["classification"]["accuracy_95_bootstrap_ci"] = bootstrap_accuracy_ci(rows, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
