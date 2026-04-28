"""Visualize what the autoencoder understands in its latent space.

Produces:
  1. t-SNE of latent vectors colored by digit class
  2. PCA of latent vectors colored by digit class
  3. Top-k activated latent dimensions per class (heatmap)
  4. Latent interpolation between pairs of digits — decoded into 8x8 patches
  5. Per-dimension sweep: vary one latent dim while fixing the rest, decode
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import torch
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "MnistSimpleCNN" / "code"))

from models.modelM5 import ModelM5
from class_prediction import predict_proba, SEED, VAL_SIZE, BATCH_SIZE as CLS_BATCH
from pixel_prediction import ConditionalAutoencoder, LATENT_DIM

DATA_DIR = ROOT / "data"
OUT_DIR = Path(__file__).resolve().parent / "latent_space"
OUT_DIR.mkdir(exist_ok=True)

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

# ── Load data (val split) ────────────────────────────────────────────────────
print("Loading data...")
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

# ── Load models ──────────────────────────────────────────────────────────────
print("Loading models...")
classifier = ModelM5()
classifier.load_state_dict(torch.load(ROOT / "donut_classifier.pth", map_location=device, weights_only=True))
classifier.to(device).eval()

model = ConditionalAutoencoder()
model.load_state_dict(torch.load(ROOT / "pixel_predictor.pth", map_location=device, weights_only=True))
model.to(device).eval()

# ── Extract latents + class info ─────────────────────────────────────────────
print("Extracting latent representations...")
all_probs, all_latents = [], []
with torch.no_grad():
    for i in range(0, len(masked), CLS_BATCH):
        m = masked[i : i + CLS_BATCH].to(device)
        probs = predict_proba(classifier, m)
        confs = probs.max(1).values.unsqueeze(1)
        latent = model.encode(m, probs, confs)
        all_probs.append(probs.cpu())
        all_latents.append(latent.cpu())

all_probs = torch.cat(all_probs)
all_latents = torch.cat(all_latents).numpy()
all_preds = all_probs.argmax(1).numpy()
digit_labels = labels.astype(int)

COLORS = plt.cm.tab10(np.arange(10))

# ═════════════════════════════════════════════════════════════════════════════
# 1. t-SNE
# ═════════════════════════════════════════════════════════════════════════════
print("Computing t-SNE (this may take a minute)...")
tsne = TSNE(n_components=2, random_state=SEED, perplexity=40, max_iter=1000)
Z2 = tsne.fit_transform(all_latents)

fig, ax = plt.subplots(figsize=(10, 10))
for d in range(10):
    mask = digit_labels == d
    ax.scatter(Z2[mask, 0], Z2[mask, 1], c=[COLORS[d]], label=str(d),
               s=8, alpha=0.6, edgecolors="none")
ax.legend(title="Digit", markerscale=3, fontsize=9)
ax.set_title("t-SNE of latent space (colored by digit class)")
ax.set_xlabel("t-SNE 1")
ax.set_ylabel("t-SNE 2")
fig.tight_layout()
fig.savefig(OUT_DIR / "tsne.png", dpi=150)
plt.close(fig)
print(f"  -> saved tsne.png")

# ═════════════════════════════════════════════════════════════════════════════
# 2. PCA
# ═════════════════════════════════════════════════════════════════════════════
print("Computing PCA...")
pca = PCA(n_components=2, random_state=SEED)
Z_pca = pca.fit_transform(all_latents)

fig, ax = plt.subplots(figsize=(10, 10))
for d in range(10):
    mask = digit_labels == d
    ax.scatter(Z_pca[mask, 0], Z_pca[mask, 1], c=[COLORS[d]], label=str(d),
               s=8, alpha=0.6, edgecolors="none")
ax.legend(title="Digit", markerscale=3, fontsize=9)
ev = pca.explained_variance_ratio_
ax.set_title(f"PCA of latent space (PC1={ev[0]:.1%}, PC2={ev[1]:.1%})")
ax.set_xlabel("PC 1")
ax.set_ylabel("PC 2")
fig.tight_layout()
fig.savefig(OUT_DIR / "pca.png", dpi=150)
plt.close(fig)
print(f"  -> saved pca.png")

# ═════════════════════════════════════════════════════════════════════════════
# 3. Mean latent activation per class (heatmap)
# ═════════════════════════════════════════════════════════════════════════════
print("Computing per-class latent profiles...")
class_means = np.zeros((10, all_latents.shape[1]))
for d in range(10):
    class_means[d] = all_latents[digit_labels == d].mean(axis=0)

# Show top 32 most variable dimensions for readability
dim_var = class_means.var(axis=0)
top_dims = np.argsort(dim_var)[-32:][::-1]

fig, ax = plt.subplots(figsize=(14, 5))
im = ax.imshow(class_means[:, top_dims], aspect="auto", cmap="viridis")
ax.set_yticks(range(10))
ax.set_yticklabels([str(d) for d in range(10)])
ax.set_xlabel("Latent dimension (top 32 most variable)")
ax.set_ylabel("Digit class")
ax.set_title("Mean latent activation per digit class")
ax.set_xticks(range(len(top_dims)))
ax.set_xticklabels(top_dims, fontsize=6, rotation=90)
fig.colorbar(im, ax=ax, shrink=0.8)
fig.tight_layout()
fig.savefig(OUT_DIR / "class_latent_heatmap.png", dpi=150)
plt.close(fig)
print(f"  -> saved class_latent_heatmap.png")

# ═════════════════════════════════════════════════════════════════════════════
# 4. Latent interpolation between digit pairs
# ═════════════════════════════════════════════════════════════════════════════
print("Generating latent interpolations...")
rng = np.random.default_rng(SEED)
pairs = [(0, 1), (3, 8), (4, 9), (2, 7), (5, 6)]
n_steps = 10

fig, axes = plt.subplots(len(pairs), n_steps + 2, figsize=(n_steps + 2, len(pairs) * 1.2))
fig.suptitle("Latent interpolation between digit class centroids", fontsize=11, y=1.02)

with torch.no_grad():
    for row, (d_a, d_b) in enumerate(pairs):
        z_a = torch.tensor(class_means[d_a], dtype=torch.float32).unsqueeze(0).to(device)
        z_b = torch.tensor(class_means[d_b], dtype=torch.float32).unsqueeze(0).to(device)

        # Show source patches (pick a representative sample for each)
        idx_a = rng.choice(np.where(digit_labels == d_a)[0])
        idx_b = rng.choice(np.where(digit_labels == d_b)[0])

        axes[row, 0].imshow(ground_truth[idx_a, 0], cmap="gray", vmin=0, vmax=1)
        axes[row, 0].set_ylabel(f"{d_a} -> {d_b}", fontsize=8)

        for step in range(n_steps):
            alpha = step / (n_steps - 1)
            z_interp = (1 - alpha) * z_a + alpha * z_b
            patch = model.decode(z_interp).cpu().numpy()[0, 0]
            axes[row, step + 1].imshow(patch, cmap="gray", vmin=0, vmax=1)
            if row == 0:
                axes[row, step + 1].set_title(f"{alpha:.1f}", fontsize=7)

        axes[row, -1].imshow(ground_truth[idx_b, 0], cmap="gray", vmin=0, vmax=1)

        for col in range(n_steps + 2):
            axes[row, col].axis("off")

# Label first and last columns
axes[0, 0].set_title("src", fontsize=7)
axes[0, -1].set_title("dst", fontsize=7)

fig.tight_layout()
fig.savefig(OUT_DIR / "latent_interpolation.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  -> saved latent_interpolation.png")

# ═════════════════════════════════════════════════════════════════════════════
# 5. Single-dimension sweeps from the mean latent
# ═════════════════════════════════════════════════════════════════════════════
print("Generating single-dimension sweeps...")
global_mean = all_latents.mean(axis=0)
global_std = all_latents.std(axis=0)

# Pick 8 dimensions with highest variance (most informative)
sweep_dims = np.argsort(global_std)[-8:][::-1]
n_sweep = 9  # number of steps per dimension
sweep_range = np.linspace(-2, 2, n_sweep)  # in units of std

fig, axes = plt.subplots(len(sweep_dims), n_sweep, figsize=(n_sweep, len(sweep_dims) * 1.2))
fig.suptitle("Single latent dimension sweeps (-2σ to +2σ from mean)", fontsize=11, y=1.02)

with torch.no_grad():
    for row, dim in enumerate(sweep_dims):
        for col, sigma in enumerate(sweep_range):
            z = global_mean.copy()
            z[dim] = global_mean[dim] + sigma * global_std[dim]
            z_t = torch.tensor(z, dtype=torch.float32).unsqueeze(0).to(device)
            patch = model.decode(z_t).cpu().numpy()[0, 0]
            axes[row, col].imshow(patch, cmap="gray", vmin=0, vmax=1)
            axes[row, col].axis("off")
            if row == 0:
                axes[row, col].set_title(f"{sigma:+.1f}σ", fontsize=7)
        axes[row, 0].set_ylabel(f"dim {dim}", fontsize=8, rotation=0, labelpad=30, va="center")

fig.tight_layout()
fig.savefig(OUT_DIR / "dimension_sweeps.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  -> saved dimension_sweeps.png")

print(f"\nAll outputs saved to {OUT_DIR}/")
