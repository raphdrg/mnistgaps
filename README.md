# mnistgaps

Predicting the missing 8x8 center patch of MNIST digit images using a two-stage deep learning pipeline.

Given an MNIST image with the center pixels (rows 10-17, cols 10-17) zeroed out, the model predicts what those 64 pixels should be.

## How it works

The pipeline has two stages that run sequentially:

**Stage 1 - Digit Classification:** A CNN classifier (`ModelM5`) looks at the outer "donut" of the image and predicts which digit (0-9) it is. This gives the pixel predictor a strong prior about what the center should look like.

**Stage 2 - Pixel Prediction:** A conditional autoencoder takes three inputs — the masked image, the predicted class probabilities, and the classifier's confidence — and outputs the predicted 8x8 center patch.

```
masked image (28x28)  ──> [Donut Classifier] ──> class probs (10) + confidence (1)
         │                                                │
         └──────────────────> [Conditional Autoencoder] <──┘
                                       │
                                predicted 8x8 patch
```

## Repository structure

```
mnistgaps/
├── main.py                    # Full pipeline: train both stages + predict test set
├── class_prediction.py        # Stage 1: train the donut classifier standalone
├── pixel_prediction.py        # Stage 2: train the pixel predictor standalone
├── generate_labels.py         # Generate pseudo-labels using standard MNIST
│
├── data/
│   ├── train_data.npz         # Training images (60k, 1x28x28, uint8)
│   ├── test_data.npz          # Test images (10k, 1x28x28, uint8, center zeroed)
│   ├── train_labels.npy       # Pseudo-labels from generate_labels.py
│   ├── train_label_confs.npy  # Classifier confidence for each label
│   └── template_solution.py   # Original assignment template
│
├── sampling/                  # Visualization & evaluation scripts
│   ├── visualize_samples.py             # View raw training/test images
│   ├── visualize_labels.py              # View predicted digit labels
│   ├── visualize_predictions.py         # View classifier predictions on donuts
│   ├── visualize_pixel_predictions.py   # Evaluate pixel predictions (MSE, best/worst)
│   └── visualize_latent_space.py        # Explore autoencoder latent space (t-SNE, PCA, etc.)
│
└── MnistSimpleCNN/            # Git submodule with CNN model definitions
    └── code/
        ├── models/modelM5.py  # 5-layer CNN used as the donut classifier
        └── ema.py             # Exponential moving average for training
```

## Setup

```bash
# Clone with submodule
git clone --recurse-submodules <repo-url>
cd mnistgaps

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install numpy torch torchvision matplotlib tqdm scikit-learn
```

Place `train_data.npz` and `test_data.npz` in the `data/` folder.

## Usage

### Run the full pipeline (train + predict)

```bash
python main.py
```

This will:
1. Train the donut classifier (50 epochs)
2. Train the pixel predictor (100 epochs)
3. Generate predictions on the test set
4. Save output to `submit_this_test_data_output.npz`

Saved model weights: `donut_classifier.pth` and `pixel_predictor.pth`.

### Train stages separately

If you want to iterate on one stage without retraining the other:

```bash
# Step 0: Generate pseudo-labels (only needed once)
python generate_labels.py

# Step 1: Train the donut classifier
python class_prediction.py

# Step 2: Train the pixel predictor (requires donut_classifier.pth)
python pixel_prediction.py
```

### Evaluate and visualize

All visualization scripts save images to `sampling/sample_images/` or `sampling/latent_space/`.

```bash
# View raw data samples
python sampling/visualize_samples.py

# Evaluate pixel predictions — MSE distribution, best/worst examples
python sampling/visualize_pixel_predictions.py

# Explore latent space — t-SNE, PCA, class heatmap, interpolations, dimension sweeps
python sampling/visualize_latent_space.py
```

## Model details

### Donut Classifier (ModelM5)

- 5 convolutional layers (5x5 kernels), BatchNorm, ReLU
- Trained with EMA (decay=0.999) and ExponentialLR scheduler
- Input: masked 28x28 image | Output: log-softmax over 10 classes

### Conditional Autoencoder

- **Encoder:** 5-layer CNN (3x3 kernels, BatchNorm) compresses masked image to 256-dim, fused with class probs (10) + confidence (1) = 267-dim, then MLP to 256-dim latent
- **Decoder:** MLP (256 -> 512 -> 1024 -> 64) with sigmoid output, reshaped to 8x8
- **Loss:** MSE on the center patch
- **Regularization:** L2 weight decay (1e-4) via Adam optimizer

## Output format

The submission file `submit_this_test_data_output.npz` contains a `data` array of shape `(10000, 1, 28, 28)` in uint8. Only the center 8x8 region `[:, :, 10:18, 10:18]` contains predictions; the rest is zero.
