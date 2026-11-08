"""Pre-registered repeated-seed FER2013 MobileNetV3 benchmark.

This script uses the official FER2013 test folder only after selecting each
checkpoint on a stratified validation split. It exports every seed's test
predictions and aggregate uncertainty statistics; it does not fabricate data.
"""
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

SEEDS = [20261107, 20261108, 20261109]
EPOCHS = 25
BATCH_SIZE = 128
LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
INPUT, OUT = Path("/kaggle/input"), Path("/kaggle/working/affectiveecho_robustness")
OUT.mkdir(parents=True, exist_ok=True)


def find_root():
    for candidate in INPUT.rglob("train"):
        if candidate.is_dir() and all((candidate / label).is_dir() for label in LABELS):
            return candidate.parent
    raise FileNotFoundError("FER2013 train/<class> layout was not found")


ROOT = find_root()


def rows_for(partition):
    return [(path, label) for label, name in enumerate(LABELS)
            for path in sorted((ROOT / partition / name).glob("*.jpg"))]


ALL_TRAIN, TEST = rows_for("train"), rows_for("test")
assert len(ALL_TRAIN) == 28709 and len(TEST) == 7178, (len(ALL_TRAIN), len(TEST))

TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(12, translate=(0.05, 0.05)),
    transforms.Grayscale(num_output_channels=3), transforms.ToTensor(),
    transforms.Normalize((0.485,) * 3, (0.229,) * 3),
])
EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)), transforms.Grayscale(num_output_channels=3), transforms.ToTensor(),
    transforms.Normalize((0.485,) * 3, (0.229,) * 3),
])


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
    return accuracy_score(truth, predicted), truth, predicted, rows


def metrics(truth, predicted):
    precision, recall, f1, _ = precision_recall_fscore_support(truth, predicted, average="macro", zero_division=0)
    return {"accuracy": float(accuracy_score(truth, predicted)), "macro_precision": float(precision),
            "macro_recall": float(recall), "macro_f1": float(f1), "cohen_kappa": float(cohen_kappa_score(truth, predicted))}


device = torch.device("cuda")
runs = []
for seed in SEEDS:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    ids = list(range(len(ALL_TRAIN)))
    train_ids, validation_ids = train_test_split(ids, test_size=0.10, random_state=seed, stratify=[label for _, label in ALL_TRAIN])
    train_rows = [ALL_TRAIN[i] for i in train_ids]; validation_rows = [ALL_TRAIN[i] for i in validation_ids]
    loaders = {
        "train": DataLoader(FER(train_rows, TRAIN_TRANSFORM), BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True),
        "validation": DataLoader(FER(validation_rows, EVAL_TRANSFORM), BATCH_SIZE, num_workers=4, pin_memory=True),
        "test": DataLoader(FER(TEST, EVAL_TRANSFORM), BATCH_SIZE, num_workers=4, pin_memory=True),
    }
    model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(LABELS)); model.to(device)
    counts = Counter(label for _, label in train_rows)
    weights = torch.tensor([len(train_rows) / (len(LABELS) * counts[i]) for i in range(len(LABELS))], device=device)
    loss_fn = nn.CrossEntropyLoss(weight=weights); optimiser = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, EPOCHS)
    best_validation, best_path, history = -1.0, OUT / f"seed_{seed}_best.pt", []
    for epoch in range(1, EPOCHS + 1):
        model.train(); losses = []
        for images, labels, _ in loaders["train"]:
            optimiser.zero_grad(); loss = loss_fn(model(images.to(device)), labels.to(device)); loss.backward(); optimiser.step(); losses.append(loss.item())
        validation_accuracy, _, _, _ = evaluate(model, loaders["validation"], device); scheduler.step()
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_accuracy": float(validation_accuracy)})
        print(json.dumps({"seed": seed, **history[-1]}), flush=True)
        if validation_accuracy > best_validation:
            best_validation = validation_accuracy; torch.save(model.state_dict(), best_path)
    model.load_state_dict(torch.load(best_path, map_location=device))
    _, truth, predicted, predictions = evaluate(model, loaders["test"], device, export=True)
    result = {"seed": seed, "n_train": len(train_rows), "n_validation": len(validation_rows), "n_test": len(TEST),
              "best_validation_accuracy": float(best_validation), "test": metrics(truth, predicted), "history": history}
    with (OUT / f"seed_{seed}_test_predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "y_true", "y_pred"]); writer.writeheader(); writer.writerows(predictions)
    (OUT / f"seed_{seed}_metrics.json").write_text(json.dumps(result, indent=2)); runs.append(result)

summary = {key: {"mean": float(np.mean([run["test"][key] for run in runs])), "std": float(np.std([run["test"][key] for run in runs], ddof=1))}
           for key in ["accuracy", "macro_precision", "macro_recall", "macro_f1", "cohen_kappa"]}
provenance = {"dataset": "msambare/fer2013", "architecture": "MobileNetV3-Large", "seeds": SEEDS, "epochs": EPOCHS,
              "device": torch.cuda.get_device_name(0), "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(OUT / "summary.json").write_text(json.dumps({"runs": runs, "aggregate": summary, "provenance": provenance}, indent=2))
print(json.dumps({"aggregate": summary, "provenance": provenance}, indent=2), flush=True)
