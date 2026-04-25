"""Sample images alongside their pseudo-labels to visually verify label quality."""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_DIR = Path(__file__).resolve().parent / "sample_images"
OUT_DIR.mkdir(exist_ok=True)

N_ROWS, N_COLS = 5, 8  # 40 images total

images = np.load(DATA_DIR / "train_data.npz")["data"]  # (60000,1,28,28)
labels = np.load(DATA_DIR / "train_labels.npy")          # (60000,)

rng = np.random.default_rng()

# --- Random sample grid ---
idx = rng.choice(len(images), N_ROWS * N_COLS, replace=False)

fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(N_COLS * 1.5, N_ROWS * 1.8))
fig.suptitle("Random samples with pseudo-labels", fontsize=14)
for ax, i in zip(axes.flat, idx):
    ax.imshow(images[i, 0], cmap="gray", vmin=0, vmax=255)
    ax.set_title(f"label={labels[i]}", fontsize=9)
    ax.axis("off")
fig.tight_layout()
fig.savefig(OUT_DIR / "label_check_random.png", dpi=150)
plt.close(fig)

# --- Per-digit grid: 4 examples per digit ---
fig, axes = plt.subplots(10, 4, figsize=(6, 18))
fig.suptitle("4 samples per digit class", fontsize=14)
for digit in range(10):
    digit_idx = np.where(labels == digit)[0]
    chosen = rng.choice(digit_idx, 4, replace=False)
    for col, i in enumerate(chosen):
        ax = axes[digit, col]
        ax.imshow(images[i, 0], cmap="gray", vmin=0, vmax=255)
        if col == 0:
            ax.set_ylabel(f"{digit}", fontsize=12, rotation=0, labelpad=20)
        ax.axis("off")
fig.tight_layout()
fig.savefig(OUT_DIR / "label_check_per_digit.png", dpi=150)
plt.close(fig)

print(f"Saved to {OUT_DIR / 'label_check_random.png'}")
print(f"Saved to {OUT_DIR / 'label_check_per_digit.png'}")
