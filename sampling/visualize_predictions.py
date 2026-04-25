"""Visualize donut classifier predictions on masked images."""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from class_prediction import DonutClassifier, BATCH_SIZE

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_DIR = Path(__file__).resolve().parent / "sample_images"
OUT_DIR.mkdir(exist_ok=True)

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

# Load data
images = np.load(DATA_DIR / "train_data.npz")["data"]  # (60000,1,28,28)
labels = np.load(DATA_DIR / "train_labels.npy")

images_t = torch.tensor(images, dtype=torch.float32) / 255.0
masked = images_t.clone()
masked[:, :, 10:18, 10:18] = 0.0

# Load model
model = DonutClassifier()
ROOT = Path(__file__).resolve().parent.parent
model.load_state_dict(torch.load(ROOT / "donut_classifier.pth", map_location=device, weights_only=True))
model.to(device)
model.eval()

# Get predictions for all images
all_probs = []
with torch.no_grad():
    for i in range(0, len(masked), BATCH_SIZE):
        batch = masked[i : i + BATCH_SIZE].to(device)
        all_probs.append(model.predict_proba(batch).cpu())
all_probs = torch.cat(all_probs)
all_preds = all_probs.argmax(1).numpy()
all_confs = all_probs.max(1).values.numpy()

rng = np.random.default_rng()

# --- 1) Random predictions grid ---
idx = rng.choice(len(images), 24, replace=False)
fig, axes = plt.subplots(4, 6, figsize=(12, 8))
fig.suptitle("Donut classifier predictions (masked images)", fontsize=14)
for ax, i in zip(axes.flat, idx):
    ax.imshow(masked[i, 0], cmap="gray", vmin=0, vmax=1)
    correct = all_preds[i] == labels[i]
    color = "green" if correct else "red"
    ax.set_title(f"pred={all_preds[i]} ({all_confs[i]:.0%})\ntrue={labels[i]}", fontsize=8, color=color)
    ax.axis("off")
fig.tight_layout()
fig.savefig(OUT_DIR / "predictions_random.png", dpi=150)
plt.close(fig)

# --- 2) Mistakes ---
wrong = np.where(all_preds != labels)[0]
if len(wrong) > 0:
    sample_wrong = rng.choice(wrong, min(24, len(wrong)), replace=False)
    fig, axes = plt.subplots(4, 6, figsize=(12, 8))
    fig.suptitle(f"Misclassified examples ({len(wrong)} total mistakes)", fontsize=14)
    for ax, i in zip(axes.flat, sample_wrong):
        ax.imshow(masked[i, 0], cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"pred={all_preds[i]} ({all_confs[i]:.0%})\ntrue={labels[i]}", fontsize=8, color="red")
        ax.axis("off")
    for ax in axes.flat[len(sample_wrong):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "predictions_mistakes.png", dpi=150)
    plt.close(fig)

# --- 3) Confidence distribution ---
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(all_confs, bins=50, edgecolor="black", alpha=0.7)
ax.set_xlabel("Confidence")
ax.set_ylabel("Count")
ax.set_title("Prediction confidence distribution")
fig.tight_layout()
fig.savefig(OUT_DIR / "predictions_confidence.png", dpi=150)
plt.close(fig)

print(f"Overall accuracy: {(all_preds == labels).mean():.4f}")
print(f"Saved to {OUT_DIR}")
