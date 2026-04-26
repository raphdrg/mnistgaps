"""Train ModelM5 to predict digit class (0-9) from the outer donut of masked MNIST images."""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Import M5 and EMA from submodule
sys.path.insert(0, str(Path(__file__).resolve().parent / "MnistSimpleCNN" / "code"))
from models.modelM5 import ModelM5
from ema import EMA

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BATCH_SIZE = 120
EPOCHS = 50
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
# Training
# ---------------------------------------------------------------------------

def train(model, train_x, train_y, val_x, val_y):
    model.to(device)
    ema = EMA(model, decay=0.999)

    train_loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=BATCH_SIZE, shuffle=True,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

    g_step = 0
    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = F.nll_loss(output, y)
            loss.backward()
            optimizer.step()
            g_step += 1
            ema(model, g_step)

            total_loss += loss.item() * x.size(0)
            correct += (output.argmax(1) == y).sum().item()
            total += x.size(0)

        scheduler.step()
        train_acc = correct / total

        # Validation with EMA weights
        model.eval()
        ema.assign(model)
        val_acc, val_loss = evaluate(model, val_x, val_y)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "donut_classifier.pth")
        ema.resume(model)

        print(
            f"Epoch {epoch + 1:>2}/{EPOCHS}  "
            f"train_loss: {total_loss / total:.4f}  train_acc: {train_acc:.4f}  "
            f"val_loss: {val_loss:.4f}  val_acc: {val_acc:.4f}  "
            f"best: {best_val_acc:.4f}"
        )

    # Load best model
    model.load_state_dict(torch.load("donut_classifier.pth", map_location=device, weights_only=True))
    return model


def evaluate(model, x, y):
    model.eval()
    with torch.no_grad():
        preds_all, loss_sum, total = [], 0.0, 0
        for i in range(0, len(x), BATCH_SIZE):
            bx = x[i : i + BATCH_SIZE].to(device)
            by = y[i : i + BATCH_SIZE].to(device)
            output = model(bx)
            loss_sum += F.nll_loss(output, by, reduction="sum").item()
            preds_all.append(output.argmax(1).cpu())
            total += bx.size(0)

        preds_all = torch.cat(preds_all)
        acc = (preds_all == y).float().mean().item()
        return acc, loss_sum / total


def predict_proba(model, x):
    """Return softmax confidence for each class."""
    model.eval()
    with torch.no_grad():
        logits = model.get_logits(x)
        return F.softmax(logits, dim=1)


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

    model = ModelM5()
    param_count = sum(p.numel() for p in model.parameters())
    print(f"ModelM5 parameters: {param_count:,}")

    model = train(model, train_x, train_y, val_x, val_y)
    print("\nSaved best model to donut_classifier.pth")

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
