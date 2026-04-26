"""Train a conditional autoencoder to predict the 8x8 center patch of masked MNIST images.

Inputs: masked image + predicted class (one-hot) + confidence
Output: 8x8 pixel values for the center patch
Loss: MSE on pixels [10:18, 10:18]
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms.v2 as T

# Import M5 for generating class predictions
sys.path.insert(0, str(Path(__file__).resolve().parent / "MnistSimpleCNN" / "code"))
from models.modelM5 import ModelM5
from class_prediction import predict_proba, SEED, VAL_SIZE

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BATCH_SIZE = 128
EPOCHS = 120
LR = 1e-3
LATENT_DIM = 128
DATA_DIR = Path("data")

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_classifier():
    """Load the trained M5 donut classifier."""
    classifier = ModelM5()
    classifier.load_state_dict(
        torch.load("donut_classifier.pth", map_location=device, weights_only=True)
    )
    classifier.to(device)
    classifier.eval()
    return classifier


def get_class_features(classifier, masked):
    """Get class probabilities and confidence from the classifier."""
    all_probs = []
    with torch.no_grad():
        for i in range(0, len(masked), BATCH_SIZE):
            batch = masked[i : i + BATCH_SIZE].to(device)
            all_probs.append(predict_proba(classifier, batch).cpu())
    all_probs = torch.cat(all_probs)
    confs = all_probs.max(1).values.unsqueeze(1)
    return all_probs, confs


def load_data():
    images = np.load(DATA_DIR / "train_data.npz")["data"]  # (60000,1,28,28) uint8
    images = torch.tensor(images, dtype=torch.float32) / 255.0

    # Train/val split — same as class_prediction.py
    rng = np.random.RandomState(SEED)
    idx = rng.permutation(len(images))
    val_idx, train_idx = idx[:VAL_SIZE], idx[VAL_SIZE:]

    train_images = images[train_idx]  # full unmasked images for augmentation
    val_images = images[val_idx]

    # Val set: precompute masked + class features (no augmentation)
    val_masked = val_images.clone()
    val_masked[:, :, 10:18, 10:18] = 0.0
    val_targets = val_images[:, :, 10:18, 10:18].clone()

    classifier = load_classifier()
    val_probs, val_confs = get_class_features(classifier, val_masked)

    return train_images, (val_masked, val_probs, val_confs, val_targets), classifier


# Augmentation applied to full images before masking
augment = T.Compose([
    T.RandomRotation(15),
    T.RandomAffine(0, translate=(0.1, 0.1)),
])


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ConditionalAutoencoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()

        # Image encoder (masked 28x28 input)
        self.img_encoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 14x14

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 7x7

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(7),  # 7x7 -> 1x1
        )

        # Fuse image features (256) + class probs (10) + confidence (1)
        fuse_dim = 256 + 10 + 1

        # Compress to latent
        self.to_latent = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(fuse_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, latent_dim),
            nn.ReLU(inplace=True),
        )

        # Decode from latent to 8x8 patch
        self.decoder = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(latent_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1 * 8 * 8),
            nn.Sigmoid(),  # output in [0, 1]
        )

    def encode(self, masked_img, class_probs, confidence):
        img_feat = self.img_encoder(masked_img)
        img_feat = img_feat.flatten(1)  # (B, 2048)
        fused = torch.cat([img_feat, class_probs, confidence], dim=1)
        latent = self.to_latent(fused)
        return latent

    def decode(self, latent):
        out = self.decoder(latent)
        return out.view(-1, 1, 8, 8)

    def forward(self, masked_img, class_probs, confidence):
        latent = self.encode(masked_img, class_probs, confidence)
        return self.decode(latent)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(model, train_images, val_data, classifier):
    model.to(device)

    # Train loader over full (unmasked) images — augmentation applied per batch
    train_loader = DataLoader(
        TensorDataset(train_images),
        batch_size=BATCH_SIZE, shuffle=True,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        total = 0

        for (full_imgs,) in train_loader:
            # Augment full images
            aug_imgs = augment(full_imgs)

            # Extract targets and mask
            targets = aug_imgs[:, :, 10:18, 10:18].clone()
            masked = aug_imgs.clone()
            masked[:, :, 10:18, 10:18] = 0.0

            # Get class predictions from classifier
            with torch.no_grad():
                probs = predict_proba(classifier, masked.to(device)).detach()
                confs = probs.max(1).values.unsqueeze(1)

            masked = masked.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            pred = model(masked, probs, confs)
            loss = F.mse_loss(pred, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * masked.size(0)
            total += masked.size(0)

        train_loss = total_loss / total

        # Validation
        val_loss = evaluate(model, val_data)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "pixel_predictor.pth")

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch + 1:>2}/{EPOCHS}  "
            f"train_mse: {train_loss:.6f}  val_mse: {val_loss:.6f}  "
            f"best: {best_val_loss:.6f}  lr: {current_lr:.1e}"
        )

    model.load_state_dict(
        torch.load("pixel_predictor.pth", map_location=device, weights_only=True)
    )
    return model


def evaluate(model, val_data):
    model.eval()
    val_masked, val_probs, val_confs, val_targets = val_data
    total_loss = 0.0
    total = 0

    with torch.no_grad():
        for i in range(0, len(val_masked), BATCH_SIZE):
            imgs = val_masked[i : i + BATCH_SIZE].to(device)
            probs = val_probs[i : i + BATCH_SIZE].to(device)
            confs = val_confs[i : i + BATCH_SIZE].to(device)
            targets = val_targets[i : i + BATCH_SIZE].to(device)

            pred = model(imgs, probs, confs)
            loss = F.mse_loss(pred, targets, reduction="sum")
            total_loss += loss.item()
            total += imgs.size(0) * 1 * 8 * 8

    return total_loss / total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(f"Using device: {device}")

    train_images, val_data, classifier = load_data()
    val_masked, val_probs, val_confs, val_targets = val_data

    print(f"Train: {train_images.shape}  Val: {val_masked.shape}")
    print(f"Target patch shape: {val_targets.shape}")

    model = ConditionalAutoencoder()
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    model = train(model, train_images, val_data, classifier)
    print(f"\nSaved best model to pixel_predictor.pth")
