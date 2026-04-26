"""Two-stage pipeline for predicting the missing 8x8 center patch of MNIST digits.

Stage 1: Donut classifier (ModelM5) — predicts digit class from the masked image
Stage 2: Conditional autoencoder — predicts center pixels conditioned on image + class

Follows the template_solution.py format.
"""

import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent / "MnistSimpleCNN" / "code"))
from models.modelM5 import ModelM5
from ema import EMA

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEED = 42
VAL_SIZE = 5000

# Stage 1 config
CLS_BATCH_SIZE = 120
CLS_EPOCHS = 50
CLS_LR = 1e-3

# Stage 2 config
PIX_BATCH_SIZE = 128
PIX_EPOCHS = 100
PIX_LR = 1e-3
LATENT_DIM = 256


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_data():
    """
    Load training and test data.

    Returns:
    - train_data_input: Tensor[N_train, 1, 28, 28] (masked — center zeroed)
    - train_data_label: Tensor[N_train, 1, 28, 28] (full original images)
    - test_data_input:  Tensor[N_test, 1, 28, 28]  (masked — center zeroed)
    """
    train_data = np.load("data/train_data.npz")["data"]  # (60000,1,28,28) uint8
    train_data = torch.tensor(train_data, dtype=torch.float32)

    test_data_input = np.load("data/test_data.npz")["data"]  # (10000,1,28,28) uint8
    test_data_input = torch.tensor(test_data_input, dtype=torch.float32)

    # Labels are the full original images (we need the center patch as target)
    train_data_label = train_data.clone()

    # Inputs are masked — zero out center 8x8
    train_data_input = train_data.clone()
    train_data_input[:, :, 10:18, 10:18] = 0.0

    # Visualize a few training samples
    if True:
        Path("train_image_output").mkdir(exist_ok=True)
        for i in tqdm(range(20), desc="Plotting train images"):
            plt.subplot(1, 2, 1)
            plt.imshow(train_data_input[i].squeeze(), cmap="gray")
            plt.title("Training Input")
            plt.subplot(1, 2, 2)
            plt.title("Training Label")
            plt.imshow(train_data_label[i].squeeze(), cmap="gray")
            plt.savefig(f"train_image_output/image_{i}.png")
            plt.close()

    return train_data_input, train_data_label, test_data_input


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def predict_proba(classifier, x):
    """Return softmax probabilities from the classifier."""
    with torch.no_grad():
        logits = classifier.get_logits(x)
        return F.softmax(logits, dim=1)


class ConditionalAutoencoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()

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
            nn.MaxPool2d(7),  # 1x1
        )

        fuse_dim = 256 + 10 + 1

        self.to_latent = nn.Sequential(
            nn.Linear(fuse_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, latent_dim),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1 * 8 * 8),
            nn.Sigmoid(),
        )

    def encode(self, masked_img, class_probs, confidence):
        img_feat = self.img_encoder(masked_img).flatten(1)
        fused = torch.cat([img_feat, class_probs, confidence], dim=1)
        return self.to_latent(fused)

    def decode(self, latent):
        return self.decoder(latent).view(-1, 1, 8, 8)

    def forward(self, masked_img, class_probs, confidence):
        return self.decode(self.encode(masked_img, class_probs, confidence))


class Model(nn.Module):
    """Wraps both stages into a single model matching the template interface.

    forward() takes (B, 1, 28, 28) masked input and returns (B, 1, 28, 28)
    with predicted center patch placed at [10:18, 10:18], rest is zero.
    Output is in [0, 255] range.
    """

    def __init__(self, classifier, autoencoder):
        super().__init__()
        self.classifier = classifier
        self.autoencoder = autoencoder

    def forward(self, x):
        # x is (B, 1, 28, 28) in [0, 255] uint8-scale
        x_norm = x / 255.0

        # Stage 1: classify
        probs = predict_proba(self.classifier, x_norm)
        confs = probs.max(1).values.unsqueeze(1)

        # Stage 2: predict center patch
        patch = self.autoencoder(x_norm, probs, confs)  # (B, 1, 8, 8) in [0, 1]

        # Assemble full output in [0, 255]
        out = torch.zeros_like(x)
        out[:, :, 10:18, 10:18] = patch * 255.0
        return out


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def training(train_data_input, train_data_label, **kwargs):
    """
    Train the two-stage model.

    Args:
    - train_data_input: Tensor[N, 1, 28, 28] (masked, [0, 255] scale)
    - train_data_label: Tensor[N, 1, 28, 28] (full images, [0, 255] scale)

    Returns:
    - model: Model wrapping classifier + autoencoder
    """
    # Normalize to [0, 1] for internal training
    masked_all = train_data_input / 255.0
    full_all = train_data_label / 255.0

    # Train/val split
    rng = np.random.RandomState(SEED)
    idx = rng.permutation(len(masked_all))
    val_idx, train_idx = idx[:VAL_SIZE], idx[VAL_SIZE:]

    # ==================================================================
    # Stage 1: Train donut classifier
    # ==================================================================
    print("\n=== Stage 1: Training donut classifier ===")

    # Load pseudo-labels
    labels = np.load("data/train_labels.npy")
    labels = torch.tensor(labels, dtype=torch.long)

    classifier = ModelM5()
    classifier.to(device)
    ema = EMA(classifier, decay=0.999)

    cls_train_loader = DataLoader(
        TensorDataset(masked_all[train_idx], labels[train_idx]),
        batch_size=CLS_BATCH_SIZE, shuffle=True,
    )

    cls_optimizer = torch.optim.Adam(classifier.parameters(), lr=CLS_LR)
    cls_scheduler = torch.optim.lr_scheduler.ExponentialLR(cls_optimizer, gamma=0.98)

    g_step = 0
    best_val_acc = 0.0

    for epoch in range(CLS_EPOCHS):
        classifier.train()
        total_loss, correct, total = 0.0, 0, 0

        for x, y in tqdm(cls_train_loader, desc=f"Cls Epoch {epoch+1}", leave=False):
            x, y = x.to(device), y.to(device)
            cls_optimizer.zero_grad()
            output = classifier(x)
            loss = F.nll_loss(output, y)
            loss.backward()
            cls_optimizer.step()
            g_step += 1
            ema(classifier, g_step)

            total_loss += loss.item() * x.size(0)
            correct += (output.argmax(1) == y).sum().item()
            total += x.size(0)

        cls_scheduler.step()

        # Validate with EMA weights
        classifier.eval()
        ema.assign(classifier)
        val_correct = 0
        with torch.no_grad():
            for i in range(0, len(val_idx), CLS_BATCH_SIZE):
                bx = masked_all[val_idx[i : i + CLS_BATCH_SIZE]].to(device)
                by = labels[val_idx[i : i + CLS_BATCH_SIZE]].to(device)
                val_correct += (classifier(bx).argmax(1) == by).sum().item()
        val_acc = val_correct / len(val_idx)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(classifier.state_dict(), "donut_classifier.pth")
        ema.resume(classifier)

        print(
            f"Cls Epoch {epoch+1:>2}/{CLS_EPOCHS}  "
            f"loss: {total_loss/total:.4f}  train_acc: {correct/total:.4f}  "
            f"val_acc: {val_acc:.4f}  best: {best_val_acc:.4f}"
        )

    # Load best classifier
    classifier.load_state_dict(
        torch.load("donut_classifier.pth", map_location=device, weights_only=True)
    )
    classifier.eval()
    print(f"Best classifier val accuracy: {best_val_acc:.4f}")

    # ==================================================================
    # Stage 2: Train pixel predictor
    # ==================================================================
    print("\n=== Stage 2: Training pixel predictor ===")

    # Precompute val class features
    val_masked = masked_all[val_idx]
    val_targets = full_all[val_idx][:, :, 10:18, 10:18]
    val_probs = []
    with torch.no_grad():
        for i in range(0, len(val_masked), PIX_BATCH_SIZE):
            batch = val_masked[i : i + PIX_BATCH_SIZE].to(device)
            val_probs.append(predict_proba(classifier, batch).cpu())
    val_probs = torch.cat(val_probs)
    val_confs = val_probs.max(1).values.unsqueeze(1)

    autoencoder = ConditionalAutoencoder()
    autoencoder.to(device)

    pix_train_loader = DataLoader(
        TensorDataset(full_all[train_idx]),
        batch_size=PIX_BATCH_SIZE, shuffle=True,
    )

    pix_optimizer = torch.optim.Adam(
        autoencoder.parameters(), lr=PIX_LR, betas=(0.7, 0.95), weight_decay=1e-4
    )
    pix_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        pix_optimizer, patience=5, factor=0.5
    )

    best_val_loss = float("inf")

    for epoch in range(PIX_EPOCHS):
        autoencoder.train()
        total_loss, total = 0.0, 0

        for (full_imgs,) in tqdm(pix_train_loader, desc=f"Pix Epoch {epoch+1}", leave=False):
            targets = full_imgs[:, :, 10:18, 10:18].clone()
            masked = full_imgs.clone()
            masked[:, :, 10:18, 10:18] = 0.0

            with torch.no_grad():
                probs = predict_proba(classifier, masked.to(device)).detach()
                confs = probs.max(1).values.unsqueeze(1)

            masked = masked.to(device)
            targets = targets.to(device)

            pix_optimizer.zero_grad()
            pred = autoencoder(masked, probs, confs)
            loss = F.mse_loss(pred, targets)
            loss.backward()
            pix_optimizer.step()

            total_loss += loss.item() * masked.size(0)
            total += masked.size(0)

        train_loss = total_loss / total

        # Validation
        autoencoder.eval()
        val_loss_sum, val_total = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(val_masked), PIX_BATCH_SIZE):
                imgs = val_masked[i : i + PIX_BATCH_SIZE].to(device)
                probs = val_probs[i : i + PIX_BATCH_SIZE].to(device)
                confs = val_confs[i : i + PIX_BATCH_SIZE].to(device)
                tgt = val_targets[i : i + PIX_BATCH_SIZE].to(device)
                pred = autoencoder(imgs, probs, confs)
                val_loss_sum += F.mse_loss(pred, tgt, reduction="sum").item()
                val_total += imgs.size(0) * 1 * 8 * 8
        val_loss = val_loss_sum / val_total

        pix_scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(autoencoder.state_dict(), "pixel_predictor.pth")

        current_lr = pix_optimizer.param_groups[0]["lr"]
        print(
            f"Pix Epoch {epoch+1:>2}/{PIX_EPOCHS}  "
            f"train_mse: {train_loss:.6f}  val_mse: {val_loss:.6f}  "
            f"best: {best_val_loss:.6f}  lr: {current_lr:.1e}"
        )

    # Load best autoencoder
    autoencoder.load_state_dict(
        torch.load("pixel_predictor.pth", map_location=device, weights_only=True)
    )

    # Wrap into unified model
    model = Model(classifier, autoencoder)
    model.eval()
    model.to(device)
    return model


# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------

def testing(model, test_data_input):
    """
    Predict outputs for the test set and save submission file.

    Args:
    - model: Model (wraps classifier + autoencoder)
    - test_data_input: Tensor[N, 1, 28, 28] in [0, 255] scale
    """
    model.eval()
    model.to(device)

    with torch.no_grad():
        test_data_input = test_data_input.to(device)
        test_data_output = []
        batch_size = 128
        for i in tqdm(
            range(0, test_data_input.shape[0], batch_size),
            desc="Predicting test output",
        ):
            output = model(test_data_input[i : i + batch_size])
            test_data_output.append(output.cpu())
        test_data_output = torch.cat(test_data_output)

    assert test_data_output.shape == test_data_input.shape, (
        f"Expected shape {test_data_input.shape}, but got "
        f"{test_data_output.shape}. "
        "Please ensure the output has the correct shape."
    )

    test_data_output = test_data_output.numpy()
    save_data_clipped = np.clip(test_data_output, 0, 255)
    save_data_uint8 = save_data_clipped.astype(np.uint8)
    save_data = np.zeros_like(save_data_uint8)
    save_data[:, :, 10:18, 10:18] = save_data_uint8[:, :, 10:18, 10:18]

    np.savez_compressed("submit_this_test_data_output.npz", data=save_data)

    if True:
        Path("test_image_output").mkdir(exist_ok=True)
        for i in tqdm(range(20), desc="Plotting test images"):
            plt.subplot(1, 2, 1)
            plt.title("Test Input")
            plt.imshow(test_data_input[i].squeeze().cpu().numpy(), cmap="gray")
            plt.subplot(1, 2, 2)
            plt.imshow(test_data_output[i].squeeze(), cmap="gray")
            plt.title("Test Output")
            plt.savefig(f"test_image_output/image_{i}.png")
            plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True

    print(f"Using device: {device}")

    train_data_input, train_data_label, test_data_input = load_data()
    model = training(train_data_input, train_data_label)
    testing(model, test_data_input)


if __name__ == "__main__":
    main()
