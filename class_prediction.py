"""Train a CNN to predict digit class (0-9) from the outer donut of masked MNIST images."""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BATCH_SIZE = 128
EPOCHS = 20
LR = 1e-3
SEED = 42
VAL_SIZE = 5000
DATA_DIR = Path("data")

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_data():
    images = np.load(DATA_DIR / "train_data.npz")["data"]  # (60000,1,28,28) uint8
    labels = np.load(DATA_DIR / "train_labels.npy")          # (60000,) int

    images = torch.tensor(images, dtype=torch.float32) / 255.0
    labels = torch.tensor(labels, dtype=torch.long)

    # Mask out center 8x8 to create the "donut"
    images[:, :, 10:18, 10:18] = 0.0

    # Random train/val split
    rng = np.random.RandomState(SEED)
    idx = rng.permutation(len(images))
    val_idx, train_idx = idx[:VAL_SIZE], idx[VAL_SIZE:]

    return (images[train_idx], labels[train_idx],
            images[val_idx], labels[val_idx])


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class DonutClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                # 14x14

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                # 7x7

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),        # 1x1
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

    def predict_proba(self, x):
        """Return softmax confidence for each class."""
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=1)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(model, train_x, train_y, val_x, val_y):
    model.to(device)

    train_loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=BATCH_SIZE, shuffle=True,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)

        train_acc = correct / total

        # Validation
        val_acc, val_loss = evaluate(model, val_x, val_y, criterion)

        print(
            f"Epoch {epoch + 1:>2}/{EPOCHS}  "
            f"train_loss: {total_loss / total:.4f}  train_acc: {train_acc:.4f}  "
            f"val_loss: {val_loss:.4f}  val_acc: {val_acc:.4f}"
        )


def evaluate(model, x, y, criterion=None):
    model.eval()
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        preds_all, loss_sum, total = [], 0.0, 0
        for i in range(0, len(x), BATCH_SIZE):
            bx = x[i : i + BATCH_SIZE].to(device)
            by = y[i : i + BATCH_SIZE].to(device)
            logits = model(bx)
            loss_sum += criterion(logits, by).item() * bx.size(0)
            preds_all.append(logits.argmax(1).cpu())
            total += bx.size(0)

        preds_all = torch.cat(preds_all)
        acc = (preds_all == y).float().mean().item()
        return acc, loss_sum / total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(f"Using device: {device}")

    train_x, train_y, val_x, val_y = load_data()
    print(f"Train: {train_x.shape}  Val: {val_x.shape}")
    print(f"Label distribution (train): {torch.bincount(train_y, minlength=10).tolist()}")
    print(f"Label distribution (val):   {torch.bincount(val_y, minlength=10).tolist()}")

    model = DonutClassifier()
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    train(model, train_x, train_y, val_x, val_y)

    # Save model
    torch.save(model.state_dict(), "donut_classifier.pth")
    print("\nSaved model to donut_classifier.pth")

    # Show per-class accuracy on validation set
    model.eval()
    with torch.no_grad():
        all_preds = []
        for i in range(0, len(val_x), BATCH_SIZE):
            bx = val_x[i : i + BATCH_SIZE].to(device)
            all_preds.append(model(bx).argmax(1).cpu())
        all_preds = torch.cat(all_preds)

    print("\nPer-class validation accuracy:")
    for c in range(10):
        mask = val_y == c
        if mask.sum() > 0:
            acc = (all_preds[mask] == c).float().mean().item()
            print(f"  Digit {c}: {acc:.4f}  ({mask.sum().item()} samples)")
