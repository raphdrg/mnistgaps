"""Train a U-Net to predict the missing 8x8 center patch of MNIST digits."""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BATCH_SIZE = 128
EPOCHS = 30
LR = 1e-3
SEED = 42
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
    train_full = np.load(DATA_DIR / "train_data.npz")["data"]  # (60000,1,28,28) uint8
    test_input = np.load(DATA_DIR / "test_data.npz")["data"]   # (10000,1,28,28) uint8

    train_full = torch.tensor(train_full, dtype=torch.float32) / 255.0
    test_input = torch.tensor(test_input, dtype=torch.float32) / 255.0

    # Create masked inputs: zero out center 8x8
    train_input = train_full.clone()
    train_input[:, :, 10:18, 10:18] = 0.0

    # Labels: just the center patch
    train_label = train_full[:, :, 10:18, 10:18].clone()

    return train_input, train_label, test_input


# ---------------------------------------------------------------------------
# Model — small U-Net
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.enc1 = ConvBlock(1, 32)
        self.enc2 = ConvBlock(32, 64)
        self.enc3 = ConvBlock(64, 128)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(128, 256)

        # Decoder
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = ConvBlock(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = ConvBlock(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = ConvBlock(64, 32)

        # Output: predict only the 8x8 center patch
        self.head = nn.Sequential(
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Pad 28x28 -> 32x32 so pooling divides evenly
        x = nn.functional.pad(x, (2, 2, 2, 2), mode="reflect")

        e1 = self.enc1(x)          # 32x32
        e2 = self.enc2(self.pool(e1))  # 16x16
        e3 = self.enc3(self.pool(e2))  # 8x8

        b = self.bottleneck(self.pool(e3))  # 4x4

        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))  # 8x8
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))  # 16x16
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))  # 32x32

        out = self.head(d1)

        # Remove padding: 32x32 -> 28x28
        out = out[:, :, 2:30, 2:30]

        # Return only the center 8x8 patch
        return out[:, :, 10:18, 10:18]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(model, train_input, train_label):
    model.to(device)
    model.train()

    dataset = TensorDataset(train_input, train_label)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    for epoch in range(EPOCHS):
        total_loss = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)

        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch + 1:>2}/{EPOCHS}  loss: {avg_loss:.6f}")


# ---------------------------------------------------------------------------
# Testing / submission
# ---------------------------------------------------------------------------

def test(model, test_input):
    model.to(device)
    model.eval()

    preds = []
    with torch.no_grad():
        for i in range(0, len(test_input), BATCH_SIZE):
            x = test_input[i : i + BATCH_SIZE].to(device)
            pred = model(x)
            preds.append(pred.cpu())
    preds = torch.cat(preds)  # (N, 1, 8, 8) in [0,1]

    # Build full output image (zeros everywhere, predicted patch in center)
    output = np.zeros((len(test_input), 1, 28, 28), dtype=np.uint8)
    center = (preds.numpy() * 255.0).clip(0, 255).astype(np.uint8)
    output[:, :, 10:18, 10:18] = center

    out_path = "submit_this_test_data_output.npz"
    np.savez_compressed(out_path, data=output)
    print(f"Saved submission to {out_path}  shape={output.shape}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(f"Using device: {device}")

    train_input, train_label, test_input = load_data()
    print(f"Train: {train_input.shape}  Test: {test_input.shape}")

    model = UNet()
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    train(model, train_input, train_label)
    test(model, test_input)
