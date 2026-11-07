"""Validate an image-data manifest before any model training begins."""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.vision.emotion_backbone import EMOTION_LABELS

REQUIRED = {"image_path", "split", "label", "valence", "arousal"}
SPLITS = {"train", "validation", "test"}


def validate_manifest(path):
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        if not REQUIRED.issubset(fields):
            raise ValueError(f"Missing columns: {sorted(REQUIRED - fields)}")
        rows = list(reader)
    if not rows:
        raise ValueError("Manifest is empty.")
    seen, split_counts, label_counts, subject_splits = set(), Counter(), Counter(), defaultdict(set)
    for number, row in enumerate(rows, start=2):
        sample = row["image_path"].strip()
        if not sample or sample in seen: raise ValueError(f"Duplicate or empty image_path at row {number}: {sample}")
        seen.add(sample)
        if row["split"] not in SPLITS: raise ValueError(f"Invalid split at row {number}: {row['split']}")
        if row["label"] not in EMOTION_LABELS: raise ValueError(f"Invalid label at row {number}: {row['label']}")
        try:
            valence, arousal = float(row["valence"]), float(row["arousal"])
        except ValueError as exc:
            raise ValueError(f"Non-numeric valence/arousal at row {number}") from exc
        if not (-1 <= valence <= 1 and -1 <= arousal <= 1):
            raise ValueError(f"Valence/arousal must be in [-1, 1] at row {number}")
        split_counts[row["split"]] += 1; label_counts[row["label"]] += 1
        if "subject_id" in fields and row["subject_id"].strip(): subject_splits[row["subject_id"].strip()].add(row["split"])
    missing = SPLITS - set(split_counts)
    if missing: raise ValueError(f"Missing required split(s): {sorted(missing)}")
    leakage = sorted(subject for subject, splits in subject_splits.items() if len(splits) > 1)
    if leakage: raise ValueError(f"Subject leakage across splits: {leakage[:10]}")
    return {"manifest": str(path), "n_samples": len(rows), "split_counts": dict(split_counts),
            "label_counts": dict(label_counts), "subject_disjointness_checked": "subject_id" in fields}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate a training manifest and prevent split leakage.")
    parser.add_argument("--manifest", required=True, type=Path); parser.add_argument("--output", type=Path)
    args = parser.parse_args(); result = validate_manifest(args.manifest)
    if args.output: args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
