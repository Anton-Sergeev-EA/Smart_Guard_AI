"""
STARGUARD AI — обучение CNN-классификатора дефектов поверхности металла.

Компактная свёрточная сеть (4 conv-блока), обучается на synthetic_defects/.
Достаточно лёгкая, чтобы обучаться на CPU за пару минут, но это настоящее
обучение настоящей нейросети (PyTorch), а не заглушка.
"""
import os
import json
import random
from glob import glob

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from generate_dataset import CLASSES, SEVERITY, IMG_SIZE

torch.manual_seed(0)
random.seed(0)

HERE = os.path.dirname(__file__)
DATA_DIR = os.path.join(HERE, "..", "data", "synthetic_defects")
MODEL_DIR = os.path.join(HERE, "..", "data", "models")
os.makedirs(MODEL_DIR, exist_ok=True)


class DefectDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("L")
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = (arr - 0.5) / 0.25
        tensor = torch.from_numpy(arr).unsqueeze(0)
        return tensor, label


def build_samples():
    samples = []
    for ci, cls in enumerate(CLASSES):
        for p in sorted(glob(os.path.join(DATA_DIR, cls, "*.png"))):
            samples.append((p, ci))
    random.shuffle(samples)
    n_val = int(len(samples) * 0.15)
    return samples[n_val:], samples[:n_val]


class DefectCNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),   # 128->64
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),  # 64->32
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),  # 32->16
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.AdaptiveAvgPool2d(4),  # ->4x4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total


def main():
    device = torch.device("cpu")
    train_s, val_s = build_samples()
    print(f"train={len(train_s)} val={len(val_s)}")
    train_loader = DataLoader(DefectDataset(train_s), batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(DefectDataset(val_s), batch_size=64, shuffle=False, num_workers=0)

    model = DefectCNN(len(CLASSES)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=6, gamma=0.5)
    loss_fn = nn.CrossEntropyLoss()

    epochs = 14
    history = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()
            total_loss += loss.item() * x.size(0)
        sched.step()
        train_loss = total_loss / len(train_s)
        val_acc = evaluate(model, val_loader, device)
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_acc": val_acc})
        print(f"epoch {epoch+1}/{epochs}  loss={train_loss:.4f}  val_acc={val_acc:.4f}")

    final_acc = evaluate(model, val_loader, device)
    torch.save({"state_dict": model.state_dict(), "classes": CLASSES, "img_size": IMG_SIZE},
               os.path.join(MODEL_DIR, "defect_cnn.pt"))
    with open(os.path.join(MODEL_DIR, "training_report.json"), "w", encoding="utf-8") as f:
        json.dump({"classes": CLASSES, "severity": SEVERITY, "final_val_acc": final_acc, "history": history}, f, ensure_ascii=False, indent=2)
    print(f"Модель сохранена. Итоговая точность на валидации: {final_acc:.4f}")


if __name__ == "__main__":
    main()
