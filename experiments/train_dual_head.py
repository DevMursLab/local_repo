"""Train AffectiveDualHeadNet from a manifest of licensed face images.

Required manifest columns: image_path,split,label,valence,arousal.  `split` must
be train, validation, or test; test is never used during fitting.
"""
import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.vision.emotion_backbone import AffectiveDualHeadNet, EMOTION_LABELS
from experiments.validate_manifest import validate_manifest


class ManifestDataset(Dataset):
    def __init__(self, manifest, split):
        with Path(manifest).open(newline="", encoding="utf-8") as handle:
            rows = [row for row in csv.DictReader(handle) if row["split"] == split]
        required = {"image_path", "split", "label", "valence", "arousal"}
        if not rows or not required.issubset(rows[0]):
            raise ValueError(f"{split} manifest rows must contain {sorted(required)}")
        self.rows = rows
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)), transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def __len__(self): return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image_path = Path(row["image_path"])
        if not image_path.is_file(): raise FileNotFoundError(image_path)
        if row["label"] not in EMOTION_LABELS: raise ValueError(f"Unknown label: {row['label']}")
        image = Image.open(image_path).convert("RGB")
        target = torch.tensor([float(row["valence"]), float(row["arousal"])], dtype=torch.float32)
        return self.transform(image), EMOTION_LABELS.index(row["label"]), target, row["image_path"]


def evaluate(model, loader, device):
    model.eval(); correct = total = 0; predictions = []
    with torch.no_grad():
        for images, labels, va, ids in loader:
            logits, predicted_va = model(images.to(device))
            classes = logits.argmax(dim=1).cpu()
            correct += (classes == labels).sum().item(); total += len(labels)
            for sample_id, truth, pred, target_va, output_va in zip(ids, labels, classes, va, predicted_va.cpu()):
                predictions.append({"sample_id": sample_id, "y_true": EMOTION_LABELS[truth], "y_pred": EMOTION_LABELS[pred],
                                    "valence_true": float(target_va[0]), "arousal_true": float(target_va[1]),
                                    "valence_pred": float(output_va[0]), "arousal_pred": float(output_va[1])})
    return correct / max(total, 1), predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True); parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--epochs", type=int, default=50); parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3); parser.add_argument("--seed", type=int, default=20261107)
    parser.add_argument("--classification-only", action="store_true", help="Train no valence/arousal loss; required for datasets such as FER2013.")
    parser.add_argument("--class-balanced", action="store_true", help="Use inverse-frequency class weights from the training split.")
    args = parser.parse_args()
    manifest_audit = validate_manifest(args.manifest)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"; out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    train = DataLoader(ManifestDataset(args.manifest, "train"), batch_size=args.batch_size, shuffle=True)
    valid = DataLoader(ManifestDataset(args.manifest, "validation"), batch_size=args.batch_size)
    test = DataLoader(ManifestDataset(args.manifest, "test"), batch_size=args.batch_size)
    model = AffectiveDualHeadNet(pretrained=True).to(device); optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    weights = None
    if args.class_balanced:
        counts = np.bincount([EMOTION_LABELS.index(row["label"]) for row in train.dataset.rows], minlength=len(EMOTION_LABELS))
        weights = (counts.sum() / (len(EMOTION_LABELS) * np.maximum(counts, 1))).astype(np.float32)
    class_loss, regression_loss = nn.CrossEntropyLoss(weight=None if weights is None else torch.tensor(weights, device=device)), nn.MSELoss(); best_accuracy = -1.0
    for _ in range(args.epochs):
        model.train()
        for images, labels, va, _ in train:
            logits, predicted_va = model(images.to(device)); loss = class_loss(logits, labels.to(device))
            if not args.classification_only:
                loss = loss + 0.5 * regression_loss(predicted_va, va.to(device))
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        accuracy, _ = evaluate(model, valid, device)
        if accuracy > best_accuracy:
            best_accuracy = accuracy; torch.save({"model": model.state_dict(), "labels": EMOTION_LABELS, "seed": args.seed}, out / "best_model.pt")
    checkpoint = out / "best_model.pt"; model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)["model"])
    _, predictions = evaluate(model, test, device)
    with (out / "test_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=predictions[0].keys()); writer.writeheader(); writer.writerows(predictions)
    provenance = {"manifest": str(args.manifest), "manifest_sha256": hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
                  "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(), "seed": args.seed, "epochs": args.epochs,
                  "device": device, "classification_only": args.classification_only, "class_balanced": args.class_balanced,
                  "class_weights": None if weights is None else weights.tolist(),
                  "validation_accuracy_for_selection": best_accuracy, "manifest_audit": manifest_audit}
    (out / "training_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")


if __name__ == "__main__": main()
