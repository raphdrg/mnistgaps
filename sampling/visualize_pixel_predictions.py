"""Visualize pixel prediction results on the val set."""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "MnistSimpleCNN" / "code"))

from models.modelM5 import ModelM5
from class_prediction import predict_proba, SEED, VAL_SIZE, BATCH_SIZE as CLS_BATCH
from pixel_prediction import ConditionalAutoencoder

DATA_DIR = ROOT / "data"
OUT_DIR = Path(__file__).resolve().parent / "sample_images"
OUT_DIR.mkdir(exist_ok=True)

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

# Load data — val split only
images = np.load(DATA_DIR / "train_data.npz")["data"]
labels = np.load(DATA_DIR / "train_labels.npy")
images = torch.tensor(images, dtype=torch.float32) / 255.0

rng_split = np.random.RandomState(SEED)
idx = rng_split.permutation(len(images))
val_idx = idx[:VAL_SIZE]

images = images[val_idx]
labels = labels[val_idx]

ground_truth = images[:, :, 10:18, 10:18].clone()
masked = images.clone()
masked[:, :, 10:18, 10:18] = 0.0

# Get class predictions
classifier = ModelM5()
classifier.load_state_dict(torch.load(ROOT / "donut_classifier.pth", map_location=device, weights_only=True))
classifier.to(device)
classifier.eval()

all_probs = []
with torch.no_grad():
    for i in range(0, len(masked), CLS_BATCH):
        batch = masked[i : i + CLS_BATCH].to(device)
        all_probs.append(predict_proba(classifier, batch).cpu())
all_probs = torch.cat(all_probs)
all_confs = all_probs.max(1).values.unsqueeze(1)
all_preds = all_probs.argmax(1).numpy()

# Get pixel predictions
model = ConditionalAutoencoder()
model.load_state_dict(torch.load(ROOT / "pixel_predictor.pth", map_location=device, weights_only=True))
model.to(device)
model.eval()

pred_patches = []
with torch.no_grad():
    for i in range(0, len(masked), CLS_BATCH):
        m = masked[i : i + CLS_BATCH].to(device)
        p = all_probs[i : i + CLS_BATCH].to(device)
        c = all_confs[i : i + CLS_BATCH].to(device)
        pred_patches.append(model(m, p, c).cpu())
pred_patches = torch.cat(pred_patches)

rng = np.random.default_rng()

# --- 1) Random samples: masked | predicted center | ground truth | full reconstructed ---
sample_idx = rng.choice(len(masked), 16, replace=False)

fig, axes = plt.subplots(16, 5, figsize=(10, 32))
fig.suptitle("Pixel predictions (val set)", fontsize=14, y=1.0)
col_titles = ["Masked input", "Predicted patch", "Ground truth patch", "Reconstructed", "Ground truth full"]

for col, title in enumerate(col_titles):
    axes[0, col].set_title(title, fontsize=10)

for row, i in enumerate(sample_idx):
    # Masked input
    axes[row, 0].imshow(masked[i, 0], cmap="gray", vmin=0, vmax=1)
    axes[row, 0].set_ylabel(f"pred={all_preds[i]}\ntrue={int(labels[i])}", fontsize=7)

    # Predicted patch
    axes[row, 1].imshow(pred_patches[i, 0], cmap="gray", vmin=0, vmax=1)

    # Ground truth patch
    axes[row, 2].imshow(ground_truth[i, 0], cmap="gray", vmin=0, vmax=1)

    # Full reconstructed image
    recon = masked[i].clone()
    recon[:, 10:18, 10:18] = pred_patches[i]
    axes[row, 3].imshow(recon[0], cmap="gray", vmin=0, vmax=1)

    # Ground truth full image
    axes[row, 4].imshow(images[i, 0], cmap="gray", vmin=0, vmax=1)

    for col in range(5):
        axes[row, col].axis("off")

fig.tight_layout()
fig.savefig(OUT_DIR / "pixel_predictions_random.png", dpi=150)
plt.close(fig)

# --- 2) MSE per-sample histogram ---
per_sample_mse = F.mse_loss(pred_patches, ground_truth, reduction="none").mean(dim=(1, 2, 3)).numpy()

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(per_sample_mse, bins=50, edgecolor="black", alpha=0.7)
ax.set_xlabel("Per-sample MSE")
ax.set_ylabel("Count")
ax.set_title(f"Val set MSE distribution (mean={per_sample_mse.mean():.4f})")
ax.axvline(per_sample_mse.mean(), color="red", linestyle="--", label=f"mean={per_sample_mse.mean():.4f}")
ax.legend()
fig.tight_layout()
fig.savefig(OUT_DIR / "pixel_predictions_mse_dist.png", dpi=150)
plt.close(fig)

# --- 3) Best and worst predictions ---
sorted_idx = np.argsort(per_sample_mse)
best_idx = sorted_idx[:8]
worst_idx = sorted_idx[-8:]

fig, axes = plt.subplots(2, 8, figsize=(16, 4))
fig.suptitle("Best predictions (lowest MSE)", fontsize=12)
for col, i in enumerate(best_idx):
    recon = masked[i].clone()
    recon[:, 10:18, 10:18] = pred_patches[i]
    axes[0, col].imshow(recon[0], cmap="gray", vmin=0, vmax=1)
    axes[0, col].set_title(f"MSE={per_sample_mse[i]:.4f}", fontsize=7)
    axes[0, col].axis("off")
    axes[1, col].imshow(images[val_idx[0]:val_idx[0]+1][0, 0] if False else ground_truth[i, 0], cmap="gray", vmin=0, vmax=1)
    # Show original full image for reference
    orig = images[i].clone()
    axes[1, col].imshow(orig[0], cmap="gray", vmin=0, vmax=1)
    axes[1, col].set_title("original", fontsize=7)
    axes[1, col].axis("off")
fig.tight_layout()
fig.savefig(OUT_DIR / "pixel_predictions_best.png", dpi=150)
plt.close(fig)

fig, axes = plt.subplots(2, 8, figsize=(16, 4))
fig.suptitle("Worst predictions (highest MSE)", fontsize=12)
for col, i in enumerate(worst_idx):
    recon = masked[i].clone()
    recon[:, 10:18, 10:18] = pred_patches[i]
    axes[0, col].imshow(recon[0], cmap="gray", vmin=0, vmax=1)
    axes[0, col].set_title(f"MSE={per_sample_mse[i]:.4f}", fontsize=7)
    axes[0, col].axis("off")
    orig = images[i].clone()
    axes[1, col].imshow(orig[0], cmap="gray", vmin=0, vmax=1)
    axes[1, col].set_title("original", fontsize=7)
    axes[1, col].axis("off")
fig.tight_layout()
fig.savefig(OUT_DIR / "pixel_predictions_worst.png", dpi=150)
plt.close(fig)

print(f"Val MSE: {per_sample_mse.mean():.6f}")
print(f"Saved to {OUT_DIR}")
