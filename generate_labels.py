"""Train ModelM5 from MnistSimpleCNN on standard MNIST, then label our dataset."""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

# Import from submodule
sys.path.insert(0, str(Path(__file__).resolve().parent / "MnistSimpleCNN" / "code"))
from models.modelM5 import ModelM5
from ema import EMA

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEED = 9717236
EPOCHS = 50
BATCH_SIZE = 120
LR = 1e-3
DATA_DIR = Path("data")

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Load standard MNIST via torchvision
    # ------------------------------------------------------------------
    mnist_train = datasets.MNIST("/tmp/mnist", train=True, download=True)
    mnist_test = datasets.MNIST("/tmp/mnist", train=False, download=True)

    # Model expects [0,1] float input (it does its own (x-0.5)*2 normalization)
    X_train = mnist_train.data.float().unsqueeze(1) / 255.0
    y_train = mnist_train.targets
    X_test = mnist_test.data.float().unsqueeze(1) / 255.0
    y_test = mnist_test.targets

    train_loader = DataLoader(
        TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False
    )

    # ------------------------------------------------------------------
    # Train ModelM5
    # ------------------------------------------------------------------
    model = ModelM5().to(device)
    ema = EMA(model, decay=0.999)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"ModelM5 parameters: {param_count:,}")

    g_step = 0
    best_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        train_loss, train_correct = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = F.nll_loss(out, y)
            loss.backward()
            optimizer.step()
            g_step += 1
            ema(model, g_step)
            train_loss += loss.item() * x.size(0)
            train_correct += (out.argmax(1) == y).sum().item()

        scheduler.step()

        # Evaluate with EMA weights
        model.eval()
        ema.assign(model)
        test_correct = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(1)
                test_correct += (pred == y).sum().item()
        test_acc = test_correct / len(y_test)
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), "m5_best.pth")
        ema.resume(model)

        train_acc = train_correct / len(y_train)
        print(
            f"Epoch {epoch + 1:>3}/{EPOCHS}  "
            f"train_acc: {train_acc:.4f}  "
            f"test_acc: {test_acc:.4f}  "
            f"best: {best_acc:.4f}"
        )

    # ------------------------------------------------------------------
    # Load best model and label our dataset
    # ------------------------------------------------------------------
    model.load_state_dict(torch.load("m5_best.pth", map_location=device, weights_only=True))
    model.eval()
    print(f"\nBest MNIST test accuracy: {best_acc:.4f}")

    ours = np.load(DATA_DIR / "train_data.npz")["data"]  # (60000,1,28,28) uint8
    ours_t = torch.tensor(ours, dtype=torch.float32) / 255.0

    labels = []
    confs = []
    with torch.no_grad():
        for i in range(0, len(ours_t), BATCH_SIZE):
            batch = ours_t[i : i + BATCH_SIZE].to(device)
            logits = model.get_logits(batch)
            probs = F.softmax(logits, dim=1)
            conf, pred = probs.max(1)
            labels.append(pred.cpu())
            confs.append(conf.cpu())

    labels = torch.cat(labels).numpy()
    confs = torch.cat(confs).numpy()

    print(f"Label distribution: {np.bincount(labels, minlength=10)}")
    print(f"Confidence: min={confs.min():.4f}  mean={confs.mean():.4f}  median={np.median(confs):.4f}")
    print(f"Low confidence (<0.9): {(confs < 0.9).sum()} / {len(confs)}")

    np.save(DATA_DIR / "train_labels.npy", labels)
    np.save(DATA_DIR / "train_label_confs.npy", confs)
    print(f"\nSaved labels to {DATA_DIR / 'train_labels.npy'}")
    print(f"Saved confidences to {DATA_DIR / 'train_label_confs.npy'}")


if __name__ == "__main__":
    main()
