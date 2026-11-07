"""Create a deterministic FER2013 manifest for categorical-emotion training.

FER2013 has no valence/arousal annotations. Placeholder zeros are emitted only
to satisfy the shared manifest schema; use this manifest with
`--classification-only` and never report dimensional-affect metrics from it.
"""
import argparse
import csv
from pathlib import Path

from sklearn.model_selection import train_test_split

LABEL_MAP = {"angry": "Angry", "disgust": "Disgust", "fear": "Fear", "happy": "Happy", "neutral": "Neutral", "sad": "Sad", "surprise": "Surprise"}


def records(root, partition):
    result = []
    for source, label in LABEL_MAP.items():
        for image in sorted((root / partition / source).glob("*.jpg")):
            result.append({"image_path": str(image.resolve()), "label": label})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/external/fer2013"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/fer2013_manifest.csv"))
    parser.add_argument("--seed", type=int, default=20261107)
    args = parser.parse_args()
    train = records(args.root, "train"); test = records(args.root, "test")
    if not train or not test: raise FileNotFoundError("Expected FER2013 train/ and test/ class folders.")
    indices = list(range(len(train))); labels = [row["label"] for row in train]
    train_ids, valid_ids = train_test_split(indices, test_size=0.1, random_state=args.seed, stratify=labels)
    rows = []
    for split, ids, source in (("train", train_ids, train), ("validation", valid_ids, train), ("test", range(len(test)), test)):
        for index in ids:
            rows.append({"image_path": source[index]["image_path"], "split": split, "label": source[index]["label"], "valence": "0", "arousal": "0"})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_path", "split", "label", "valence", "arousal"]); writer.writeheader(); writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.output}; use only with --classification-only.")


if __name__ == "__main__": main()
