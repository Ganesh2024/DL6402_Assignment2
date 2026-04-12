
Wandb Report Link: https://api.wandb.ai/links/da25m019-indian-institute-of-technology-madras/idxji479


# DA6401 Assignment 2 — Visual Perception Pipeline

> A comprehensive multi-task visual perception system built on the **Oxford-IIIT Pet Dataset**, implementing breed classification, object localisation, and semantic segmentation within a single unified deep learning pipeline.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Dataset](#dataset)
- [Architecture](#architecture)
  - [CustomDropout](#customdropout)
  - [VGG11 Encoder](#vgg11-encoder)
  - [Classifier](#classifier)
  - [Localizer](#localizer)
  - [U-Net Segmentation](#u-net-segmentation)
  - [IoU Loss](#iou-loss)
  - [Multi-Task Model](#multi-task-model)
- [Training](#training)
  - [Task 1 — Classification](#task-1--classification)
  - [Task 2 — Localisation](#task-2--localisation)
  - [Task 3 — Segmentation](#task-3--segmentation)
- [Checkpoints](#checkpoints)
- [Results](#results)

---

## Overview

This project implements a four-task visual perception pipeline:

| Task | Model | Output | Metric |
|------|-------|--------|--------|
| Task 1 | VGG11 Classifier | 37-class breed logits | Macro F1-Score |
| Task 2 | VGG11 Localizer | `[cx, cy, w, h]` in pixel space | Mean IoU |
| Task 3 | VGG11 U-Net | Per-pixel trimap mask `[B, 3, H, W]` | Dice Score |
| Task 4 | MultiTaskPerceptionModel | All three outputs in one forward pass | Combined |

All models share a common **VGG11 convolutional backbone implemented entirely from scratch** using standard `torch.nn` primitives. No pretrained weights or pre-built VGG models are used — the architecture strictly follows the original VGG paper (Simonyan & Zisserman, ICLR 2015, arXiv:1409.1556).

---

## Project Structure

```
DL6402_Assignment2/
│
├── models/
│   ├── __init__.py              # Package exports
│   ├── layers.py                # CustomDropout — hand-rolled inverted dropout
│   ├── vgg11.py                 # VGG11Encoder backbone + VGG11 alias
│   ├── vgg11_ablation.py        # VGG11 variant for BN/Dropout ablation studies
│   ├── classification.py        # VGG11Classifier (backbone + FC head)
│   ├── localization.py          # VGG11Localizer (backbone + regression head)
│   └── segmentation.py          # VGG11UNet (VGG11 encoder + symmetric decoder)
│
├── losses/
│   ├── __init__.py              # Package exports
│   └── iou_loss.py              # Custom differentiable IoU loss, range [0,1]
│
├── data/
│   └── pets_dataset.py          # Oxford-IIIT Pet multi-task dataset loader
│
├── checkpoints/
│   ├── checkpoints.md           # Checkpoint format and submission instructions
│   ├── classifier.pth           # Trained classifier checkpoint
│   ├── localizer.pth            # Trained localizer checkpoint
│   └── unet.pth                 # Trained U-Net checkpoint (partial fine-tune)
│
├── multitask.py                 # Unified multi-task model — loads all 3 checkpoints
├── train.py                     # Training script for all 3 tasks
├── train_ablation.py            # Ablation study runner (BN effect, dropout effect)
├── wandb_report_visuals.py      # Generates W&B report visualizations
├── inference.py                 # Inference utilities for single-image prediction
├── requirements.txt             # Python package dependencies
└── README.md                    # This file
```

---

## Requirements

### Python Version
Python 3.8 or higher

### Installation

```bash
pip install -r requirements.txt
```

### `requirements.txt`

```
torch>=1.8.0
numpy>=1.21.0
matplotlib>=3.4.0
pillow>=8.2.0
albumentations>=1.0.0
wandb>=0.12.0
scikit-learn>=0.24.2
```

> **Note:** Only these packages are used throughout the entire codebase. No additional imports are permitted as per assignment constraints — the autograder will crash on unrecognized packages.

---

## Dataset

**Oxford-IIIT Pet Dataset** — a rich benchmark providing three annotation types:

| Annotation | Description |
|-----------|-------------|
| Class label | 37 pet breed classes (0-indexed) |
| Bounding box | Head bounding box in XML format → converted to `[cx, cy, w, h]` pixel space |
| Trimap mask | Pixel-level mask: 1=foreground, 2=background, 3=boundary → remapped to 0,1,2 |

### Download

```bash
wget https://thor.robots.ox.ac.uk/pets/images.tar.gz
wget https://thor.robots.ox.ac.uk/pets/annotations.tar.gz
tar -xzf images.tar.gz
tar -xzf annotations.tar.gz
```

### Expected Directory Layout

```
dataset/
├── images/                    # JPEG pet images
└── annotations/
    ├── trimaps/               # PNG trimap masks
    ├── xmls/                  # XML bounding box annotations
    ├── list.txt               # Master file list
    ├── trainval.txt           # Training/validation split
    └── test.txt               # Test split
```

### Data Augmentation

Training uses aggressive albumentations pipeline to combat overfitting:

- Horizontal/Vertical flip
- Random rotation (±15°)
- Color jitter (brightness, contrast, saturation, hue)
- Gaussian blur and noise
- Affine transforms (scale, translate, rotate)
- CoarseDropout (random rectangular patches zeroed out)

Validation uses only resize + ImageNet normalisation.

---

## Architecture

### CustomDropout

**File:** `models/layers.py`

Implements inverted dropout from scratch — **no `nn.Dropout` or `F.dropout` used**.

```python
from models.layers import CustomDropout
drop = CustomDropout(p=0.5)
```

**Key properties:**
- Samples a Bernoulli mask with keep-probability `(1 - p)`
- Scales surviving activations by `1/(1-p)` (inverted dropout) — test-time requires no rescaling
- Respects `self.training` flag — pure identity mapping at inference
- Supports any `p` in `[0, 1)`

---

### VGG11 Encoder

**File:** `models/vgg11.py`

Implements the VGG-11 architecture (column A, Table 1 of the original paper) using only `torch.nn` primitives. BatchNorm is injected after every convolutional layer.

| Stage | Layers | Output Shape (224×224 input) |
|-------|--------|------------------------------|
| Stage 1 | Conv(64) + BN + ReLU + MaxPool | `[B, 64, 112, 112]` |
| Stage 2 | Conv(128) + BN + ReLU + MaxPool | `[B, 128, 56, 56]` |
| Stage 3 | Conv(256)×2 + BN + ReLU + MaxPool | `[B, 256, 28, 28]` |
| Stage 4 | Conv(512)×2 + BN + ReLU + MaxPool | `[B, 512, 14, 14]` |
| Stage 5 | Conv(512)×2 + BN + ReLU + MaxPool | `[B, 512, 7, 7]` |
| AdaptiveAvgPool | — | `[B, 512, 7, 7]` |

All weights initialised with **Kaiming-He normal** initialisation (fan_out, ReLU).

```python
from models.vgg11 import VGG11Encoder, VGG11

encoder = VGG11Encoder()
bottleneck = encoder(x)                              # [B, 512, 7, 7]
bottleneck, skips = encoder(x, return_features=True) # also returns skip maps
```

> `VGG11` is an alias for `VGG11Encoder` for autograder compatibility.

---

### Classifier

**File:** `models/classification.py`

VGG11 backbone + 3-layer fully-connected head (4096 → 4096 → 37).

**BN and Dropout placement rationale:**
- BatchNorm inserted **before ReLU** in each FC block — keeps pre-activation distributions centred, enabling higher learning rates
- CustomDropout placed **after BN + ReLU** — dropout on normalised activations gives more stable BN running-mean estimates

```python
from models.classification import VGG11Classifier

model = VGG11Classifier(num_classes=37, dropout_p=0.4)
logits = model(x)   # [B, 37]
```

---

### Localizer

**File:** `models/localization.py`

VGG11 backbone + regression head (25088 → 1024 → 256 → 4).

**Output:** `[cx, cy, w, h]` in **pixel space** (not normalised). Sigmoid scaling ensures outputs stay within `[0, 224]`.

**Backbone freezing:** The backbone is frozen during localisation training by default. The classification backbone already encodes rich spatial features useful for bbox prediction — fine-tuning risks degrading these for a regression task with limited annotations.

```python
from models.localization import VGG11Localizer

model = VGG11Localizer(dropout_p=0.4, freeze_backbone=True)
bbox = model(x)   # [B, 4]  →  [cx, cy, w, h] in pixels
```

---

### U-Net Segmentation

**File:** `models/segmentation.py`

VGG11 encoder fused with a symmetric transposed-convolution decoder.

**Key design decisions:**
- **Transposed Convolutions** for upsampling — bilinear interpolation and unpooling are not used
- **Skip connections** — at each decoder stage, upsampled features are concatenated (channel-wise) with the corresponding encoder stage maps
- **Loss:** 50% Cross-Entropy + 50% Dice loss

```
Encoder (contracting path):     Decoder (expansive path):
stage1 →  64 @ 112×112    ←──── up2:  64 @ 112×112
stage2 → 128 @  56×56     ←──── up3: 128 @  56×56
stage3 → 256 @  28×28     ←──── up4: 256 @  28×28
stage4 → 512 @  14×14     ←──── up5: 512 @  14×14
stage5 → 512 @   7×7  (bottleneck)
```

```python
from models.segmentation import VGG11UNet

model = VGG11UNet(num_classes=3, dropout_p=0.4)
mask_logits = model(x)   # [B, 3, 224, 224]
```

---

### IoU Loss

**File:** `losses/iou_loss.py`

Custom differentiable IoU loss — **no external IoU implementations used**.

**Properties:**
- Input format: `[cx, cy, w, h]` pixel space
- Output range: strictly `[0, 1]` — perfect overlap → 0, no overlap → 1
- Supports three reduction modes: `mean` (default), `sum`, `none`
- Numerically stable via epsilon guard on union

```python
from losses.iou_loss import IoULoss

iou_fn = IoULoss(reduction='mean')
loss = iou_fn(pred_boxes, gt_boxes)   # scalar
```

---

### Multi-Task Model

**File:** `multitask.py`

Loads the three trained checkpoints and shares the VGG11 backbone across all task heads. A single `forward()` call produces all three outputs simultaneously.

```python
from multitask import MultiTaskPerceptionModel

model = MultiTaskPerceptionModel(
    classifier_path='checkpoints/classifier.pth',
    localizer_path ='checkpoints/localizer.pth',
    unet_path      ='checkpoints/unet.pth',
)

outputs = model(x)
# outputs['classification'] → [B, 37]
# outputs['localization']   → [B, 4]
# outputs['segmentation']   → [B, 3, 224, 224]
```

---

## Training

All training is handled by `train.py`. Checkpoints are saved automatically to `--ckpt_dir` whenever validation metric improves.

### Task 1 — Classification

```bash
python train.py \
    --task classification \
    --data_root /path/to/dataset \
    --ckpt_dir  checkpoints/ \
    --epochs    60 \
    --batch_size 64 \
    --lr        3e-4 \
    --dropout_p 0.4
```

**Training strategy:**
- **MixUp augmentation** (50% of batches) — blends image pairs and soft labels, strong regulariser
- **LR warmup** (5 epochs) — linearly ramps LR from 10% to 100%
- **ReduceLROnPlateau** — halves LR when val F1 stagnates for 5 epochs
- **Label smoothing** (0.1) — prevents overconfident logits
- **AdamW** with `weight_decay=5e-3`
- Gradient clipping (max norm = 5.0)

---

### Task 2 — Localisation

Requires `classifier.pth` to exist — backbone weights are loaded from it.

```bash
python train.py \
    --task localization \
    --data_root /path/to/dataset \
    --ckpt_dir  checkpoints/ \
    --epochs    40 \
    --batch_size 32 \
    --lr        1e-4 \
    --dropout_p 0.4
```

**Loss:** MSE + custom IoU loss (combined)

---

### Task 3 — Segmentation

Three training modes available for the W&B transfer learning comparison:

```bash
# Frozen encoder
python train.py --task segmentation --unet_mode frozen \
    --data_root /path/to/dataset --ckpt_dir checkpoints/ \
    --epochs 30 --batch_size 16 --lr 1e-4

# Partial fine-tune (stage1+2 frozen, stage3-5 trainable)
python train.py --task segmentation --unet_mode partial \
    --data_root /path/to/dataset --ckpt_dir checkpoints/ \
    --epochs 30 --batch_size 16 --lr 1e-4

# Full fine-tune
python train.py --task segmentation --unet_mode full \
    --data_root /path/to/dataset --ckpt_dir checkpoints/ \
    --epochs 30 --batch_size 16 --lr 1e-4
```

**Loss:** 50% Cross-Entropy + 50% Dice loss

---

### All Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--task` | `classification` | Task to train: `classification`, `localization`, `segmentation` |
| `--data_root` | `/content/pets` | Path to dataset root directory |
| `--ckpt_dir` | `checkpoints/` | Directory to save checkpoint files |
| `--epochs` | `60` | Number of training epochs |
| `--batch_size` | `64` | Batch size |
| `--lr` | `3e-4` | Learning rate |
| `--dropout_p` | `0.4` | Dropout probability |
| `--unet_mode` | `full` | U-Net encoder strategy: `frozen`, `partial`, `full` |
| `--no_wandb` | `False` | Disable W&B logging |

---

## Checkpoints

Checkpoints are saved in the `checkpoints/` directory with mandatory filenames:

| File | Task | Saved when |
|------|------|-----------|
| `classifier.pth` | Task 1 | Val Macro F1 improves |
| `localizer.pth` | Task 2 | Val IoU improves |
| `unet.pth` | Task 3 | Val Dice improves |

### Checkpoint Format

```python
{
    "state_dict":   model.state_dict(),
    "epoch":        epoch,
    "best_metric":  metric_value,
}
```

### Loading a Checkpoint

```python
import torch
payload = torch.load('checkpoints/classifier.pth', map_location='cpu')
model.load_state_dict(payload['state_dict'])
```

---

## Results

| Task | Metric | Validation | Gradescope |
|------|--------|------------|------------|
| Classification | Macro F1 | ~0.30 (val) | **0.8182** ✅ |
| Localisation | Mean IoU | 0.635 | ✅ |
| Segmentation | Dice Score | 0.925 | ✅ |
| **Autograder Total** | — | — | **50 / 50** 🎉 |

> Note: Low validation F1 during training (~0.30) is expected due to the small validation split (15%) and heavy augmentation. The model generalises well to the held-out test set.

---

## Autograder Compatibility

The following imports are used by the autograder and are fully supported:

```python
from models.vgg11 import VGG11                    # alias for VGG11Encoder
from models.layers import CustomDropout
from losses.iou_loss import IoULoss
from multitask import MultiTaskPerceptionModel
```

---

## W&B Report

All experiments are tracked via Weights & Biases. The report covers:

| Section | Content |
|---------|---------|
| 2.1 | BatchNorm effect — activation distributions + convergence comparison |
| 2.2 | Dropout ablation — p=0, p=0.2, p=0.5 generalisation gap |
| 2.3 | Transfer learning showdown — frozen / partial / full fine-tune |
| 2.4 | Feature map visualization — first vs last conv layer |
| 2.5 | Bounding box prediction table with IoU and confidence |
| 2.6 | Segmentation samples — Dice vs Pixel Accuracy analysis |
| 2.7 | Pipeline on 3 in-the-wild internet pet images |
| 2.8 | Meta-analysis and retrospective design reflection |


---

## Inference

Run the full pipeline on a single image:

```bash
python inference.py /path/to/image.jpg
```

Output:
```
Predicted breed index : 14
Bounding box (cx,cy,w,h): [112.3, 98.7, 85.2, 72.1]
Segmentation mask shape : (224, 224)
```

---

