"""Visualize test predictions: masked input vs predicted (filled) output."""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent

# Load data
test_images = np.load(ROOT / "data" / "test_data.npz")["data"]  # (10000, 1, 28, 28)
predictions = np.load(ROOT / "test_predictions.npz")["data"]     # (10000, 1, 28, 28)

# Build masked version (zero out center 8x8)
masked = test_images.copy()
masked[:, :, 10:18, 10:18] = 0

# Pick a random sample of images to display
rng = np.random.default_rng(42)
n_pairs = 20
indices = rng.choice(len(test_images), size=n_pairs, replace=False)

# Layout: n_pairs rows, 2 columns (masked, predicted)
fig, axes = plt.subplots(n_pairs, 2, figsize=(3, 1.5 * n_pairs))
fig.suptitle("Masked Input  vs  Predicted Output", fontsize=12, y=1.0)

for row, idx in enumerate(indices):
    axes[row, 0].imshow(masked[idx, 0], cmap="gray", vmin=0, vmax=255)
    axes[row, 0].axis("off")

    # Composite: original surround + predicted center
    composite = test_images[idx, 0].copy()
    composite[10:18, 10:18] = predictions[idx, 0, 10:18, 10:18]
    axes[row, 1].imshow(composite, cmap="gray", vmin=0, vmax=255)
    axes[row, 1].axis("off")

axes[0, 0].set_title("Masked", fontsize=10)
axes[0, 1].set_title("Predicted", fontsize=10)

fig.tight_layout()
out_path = ROOT / "sampling" / "test_predictions_visual.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved to {out_path}")
