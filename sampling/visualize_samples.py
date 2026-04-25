"""Sample and visualize images from train and test data."""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_DIR = Path(__file__).resolve().parent / "sample_images"
OUT_DIR.mkdir(exist_ok=True)

N = 16  # images to sample

train = np.load(DATA_DIR / "train_data.npz")["data"]
test = np.load(DATA_DIR / "test_data.npz")["data"]

rng = np.random.default_rng(42)
train_idx = rng.choice(len(train), N, replace=False)
test_idx = rng.choice(len(test), N, replace=False)

# --- Train grid ---
fig, axes = plt.subplots(4, 4, figsize=(8, 8))
fig.suptitle("Train samples (complete images)", fontsize=14)
for ax, i in zip(axes.flat, train_idx):
    ax.imshow(train[i, 0], cmap="gray", vmin=0, vmax=255)
    ax.set_title(f"#{i}", fontsize=8)
    ax.axis("off")
fig.tight_layout()
fig.savefig(OUT_DIR / "train_samples.png", dpi=150)
plt.close(fig)

# --- Test grid ---
fig, axes = plt.subplots(4, 4, figsize=(8, 8))
fig.suptitle("Test samples (input to model)", fontsize=14)
for ax, i in zip(axes.flat, test_idx):
    ax.imshow(test[i, 0], cmap="gray", vmin=0, vmax=255)
    ax.set_title(f"#{i}", fontsize=8)
    ax.axis("off")
fig.tight_layout()
fig.savefig(OUT_DIR / "test_samples.png", dpi=150)
plt.close(fig)

# --- Train with center mask highlighted ---
fig, axes = plt.subplots(4, 4, figsize=(8, 8))
fig.suptitle("Train samples with center 8x8 region highlighted (the gap to predict)", fontsize=12)
for ax, i in zip(axes.flat, train_idx):
    img = np.stack([train[i, 0]] * 3, axis=-1).astype(float) / 255.0
    # tint the 8x8 center region red
    img[10:18, 10:18, 0] = np.clip(img[10:18, 10:18, 0] + 0.4, 0, 1)
    img[10:18, 10:18, 1] *= 0.5
    img[10:18, 10:18, 2] *= 0.5
    ax.imshow(img)
    ax.set_title(f"#{i}", fontsize=8)
    ax.axis("off")
fig.tight_layout()
fig.savefig(OUT_DIR / "train_with_gap_highlight.png", dpi=150)
plt.close(fig)

print(f"Saved images to {OUT_DIR}")
