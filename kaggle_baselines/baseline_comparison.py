"""Fair, validation-selected FER2013 baseline comparison on one fixed split."""
import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, cohen_kappa_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

SEED, EPOCHS, BATCH_SIZE = 20261107, 25, 128
LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
INPUT, OUT = Path("/kaggle/input"), Path("/kaggle/working/affectiveecho_baselines")
OUT.mkdir(parents=True, exist_ok=True)


def locate_root():
    for candidate in INPUT.rglob("train"):
        if candidate.is_dir() and all((candidate / label).is_dir() for label in LABELS):
            return candidate.parent
    raise FileNotFoundError("FER2013 train/<class> layout was not found")


def gather(root, partition):
    return [(path, label) for label, name in enumerate(LABELS) for path in sorted((root / partition / name).glob("*.jpg"))]


class FER(Dataset):
    def __init__(self, rows, transform): self.rows, self.transform = rows, transform
    def __len__(self): return len(self.rows)
    def __getitem__(self, index):
        path, label = self.rows[index]
        return self.transform(Image.open(path).convert("L")), label, str(path)


def evaluate(model, loader, device, export=False):
    model.eval(); truth, predicted, rows = [], [], []
    with torch.no_grad():
        for images, labels, paths in loader:
            outputs = model(images.to(device)).argmax(1).cpu().tolist(); labels = labels.tolist()
            truth.extend(labels); predicted.extend(outputs)
            if export:
                rows.extend({"sample_id": p, "y_true": LABELS[y], "y_pred": LABELS[pred]}
                            for p, y, pred in zip(paths, labels, outputs))
    precision, recall, f1, _ = precision_recall_fscore_support(truth, predicted, average="macro", zero_division=0)
    return {"accuracy": float(accuracy_score(truth, predicted)), "macro_precision": float(precision),
            "macro_recall": float(recall), "macro_f1": float(f1), "cohen_kappa": float(cohen_kappa_score(truth, predicted))}, rows


def make_model(name):
    if name == "ResNet-18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT); model.fc = nn.Linear(model.fc.in_features, len(LABELS))
    elif name == "EfficientNet-B0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT); model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(LABELS))
    else: raise ValueError(name)
    return model


random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
ROOT = locate_root(); all_train, test_rows = gather(ROOT, "train"), gather(ROOT, "test")
assert len(all_train) == 28709 and len(test_rows) == 7178
ids = list(range(len(all_train)))
train_ids, val_ids = train_test_split(ids, test_size=0.10, random_state=SEED, stratify=[label for _, label in all_train])
train_rows, val_rows = [all_train[i] for i in train_ids], [all_train[i] for i in val_ids]
train_tf = transforms.Compose([transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(), transforms.RandomAffine(12, translate=(0.05, 0.05)), transforms.Grayscale(num_output_channels=3), transforms.ToTensor(), transforms.Normalize((0.485,) * 3, (0.229,) * 3)])
eval_tf = transforms.Compose([transforms.Resize((224, 224)), transforms.Grayscale(num_output_channels=3), transforms.ToTensor(), transforms.Normalize((0.485,) * 3, (0.229,) * 3)])
loaders = {"train": DataLoader(FER(train_rows, train_tf), BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True), "validation": DataLoader(FER(val_rows, eval_tf), BATCH_SIZE, num_workers=4, pin_memory=True), "test": DataLoader(FER(test_rows, eval_tf), BATCH_SIZE, num_workers=4, pin_memory=True)}
counts = Counter(label for _, label in train_rows); weights = torch.tensor([len(train_rows) / (len(LABELS) * counts[i]) for i in range(len(LABELS))], device="cuda")
results = []
for name in ["ResNet-18", "EfficientNet-B0"]:
    model = make_model(name).to("cuda"); loss_fn = nn.CrossEntropyLoss(weight=weights); optimiser = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4); scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, EPOCHS)
    best, checkpoint, history = -1.0, OUT / f"{name.lower().replace('-', '')}_best.pt", []
    for epoch in range(1, EPOCHS + 1):
        model.train(); losses = []
        for images, labels, _ in loaders["train"]:
            optimiser.zero_grad(); loss = loss_fn(model(images.to("cuda")), labels.to("cuda")); loss.backward(); optimiser.step(); losses.append(loss.item())
        validation, _ = evaluate(model, loaders["validation"], "cuda"); scheduler.step(); history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_accuracy": validation["accuracy"]})
        print(json.dumps({"architecture": name, **history[-1]}), flush=True)
        if validation["accuracy"] > best: best = validation["accuracy"]; torch.save(model.state_dict(), checkpoint)
    model.load_state_dict(torch.load(checkpoint, map_location="cuda")); test, predictions = evaluate(model, loaders["test"], "cuda", export=True)
    slug = name.lower().replace("-", "")
    with (OUT / f"{slug}_test_predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "y_true", "y_pred"]); writer.writeheader(); writer.writerows(predictions)
    result = {"architecture": name, "seed": SEED, "epochs": EPOCHS, "n_train": len(train_rows), "n_validation": len(val_rows), "n_test": len(test_rows), "best_validation_accuracy": best, "test": test, "checkpoint": checkpoint.name, "history": history}
    (OUT / f"{slug}_metrics.json").write_text(json.dumps(result, indent=2)); results.append(result)

(OUT / "summary.json").write_text(json.dumps({"results": results, "device": torch.cuda.get_device_name(0), "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, indent=2))
