"""Kaggle GPU validation-only model-selection experiment.

FER2013 has no valence/arousal labels. This run must not be used for continuous
affect, clinical, or human-study claims.
"""
import csv
import hashlib
import json
import os
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, cohen_kappa_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

SEED = 20261107
EPOCHS = 30
BATCH_SIZE = 128
LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
INPUT = Path("/kaggle/input")
OUT = Path("/kaggle/working/affectiveecho_fer2013")

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
OUT.mkdir(parents=True, exist_ok=True)

def locate_root():
    for candidate in INPUT.rglob("train"):
        if candidate.is_dir() and all((candidate / label).is_dir() for label in LABELS):
            return candidate.parent
    raise FileNotFoundError(f"No FER2013 train/<class> structure found below {INPUT}")

ROOT = locate_root()

def gather(partition):
    rows = []
    for label in LABELS:
        for path in sorted((ROOT / partition / label).glob("*.jpg")):
            rows.append((path, LABELS.index(label)))
    return rows

train_all, test_rows = gather("train"), gather("test")
assert train_all and test_rows, f"FER2013 not found at {ROOT}"
indices = list(range(len(train_all)))
train_ids, val_ids = train_test_split(indices, test_size=0.10, random_state=SEED, stratify=[label for _, label in train_all])
train_rows, val_rows = [train_all[i] for i in train_ids], [train_all[i] for i in val_ids]

transform_train = transforms.Compose([transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(), transforms.RandomAffine(12, translate=(0.05, 0.05)), transforms.Grayscale(num_output_channels=3), transforms.ToTensor(), transforms.Normalize((0.485,)*3, (0.229,)*3)])
transform_eval = transforms.Compose([transforms.Resize((224, 224)), transforms.Grayscale(num_output_channels=3), transforms.ToTensor(), transforms.Normalize((0.485,)*3, (0.229,)*3)])

class FER(Dataset):
    def __init__(self, rows, transform): self.rows, self.transform = rows, transform
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        path, label = self.rows[i]
        return self.transform(Image.open(path).convert("L")), label, str(path)

loaders = {"train": DataLoader(FER(train_rows, transform_train), BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True), "val": DataLoader(FER(val_rows, transform_eval), BATCH_SIZE, num_workers=4, pin_memory=True), "test": DataLoader(FER(test_rows, transform_eval), BATCH_SIZE, num_workers=4, pin_memory=True)}
device = torch.device("cuda")
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(LABELS)); model.to(device)
counts = Counter(label for _, label in train_rows); weights = torch.tensor([len(train_rows) / (len(LABELS) * counts[i]) for i in range(len(LABELS))], device=device)
loss_fn = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05); optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS)

def evaluate(loader, export=False):
    model.eval(); truth=[]; predicted=[]; rows=[]
    with torch.no_grad():
        for images, labels, paths in loader:
            result = model(images.to(device)).argmax(1).cpu().tolist(); labels = labels.tolist()
            truth.extend(labels); predicted.extend(result)
            if export: rows.extend({"sample_id": p, "y_true": LABELS[y], "y_pred": LABELS[yp]} for p,y,yp in zip(paths, labels, result))
    return accuracy_score(truth, predicted), rows

best = -1.0; history=[]
for epoch in range(1, EPOCHS + 1):
    model.train(); losses=[]
    for images, labels, _ in loaders["train"]:
        optimizer.zero_grad(); loss = loss_fn(model(images.to(device)), labels.to(device)); loss.backward(); optimizer.step(); losses.append(loss.item())
    val_acc, _ = evaluate(loaders["val"]); scheduler.step()
    history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_accuracy": float(val_acc)})
    print(json.dumps(history[-1]), flush=True)
    if val_acc > best:
        best = val_acc; torch.save(model.state_dict(), OUT / "best_model.pt")

model.load_state_dict(torch.load(OUT / "best_model.pt", map_location=device)); val_acc, rows = evaluate(loaders["val"], export=True)
truth, predicted = [x["y_true"] for x in rows], [x["y_pred"] for x in rows]
precision, recall, f1, _ = precision_recall_fscore_support(truth, predicted, average="macro", zero_division=0)
with (OUT / "validation_predictions.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["sample_id", "y_true", "y_pred"]); writer.writeheader(); writer.writerows(rows)
metrics = {"task": "FER2013 categorical emotion recognition only; validation-only model selection", "dataset": "msambare/fer2013", "seed": SEED, "architecture": "EfficientNet-B0", "epochs": EPOCHS, "n_train": len(train_rows), "n_validation": len(val_rows), "n_test_used": 0, "validation_accuracy": val_acc, "validation_macro_precision": precision, "validation_macro_recall": recall, "validation_macro_f1": f1, "validation_cohen_kappa": cohen_kappa_score(truth, predicted), "class_weights": weights.detach().cpu().tolist(), "device": torch.cuda.get_device_name(0), "history": history}
(OUT / "validation_metrics.json").write_text(json.dumps(metrics, indent=2))
(OUT / "provenance.json").write_text(json.dumps({"script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "torch": torch.__version__, "cuda": torch.version.cuda, "metrics_file": "validation_metrics.json", "test_set_used": False}, indent=2))
print(json.dumps(metrics, indent=2), flush=True)
