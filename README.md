# C2LNet: Cross-modality Complementarity Learning Network for Infrared and Visible Image Fusion

This repository contains the implementation of **C2LNet**, a cross-modality complementarity learning framework for Infrared and Visible Image Fusion (IVIF). The model jointly mines and fuses complementary features from infrared and visible images at both semantic and spatial levels, with feature-level adversarial training for enhanced fusion quality.

## Architecture Overview

```
Visible Image ──► Vis Branch (Conv + DRC) ──► Weighted Fusion ──► Spatial Attention ──► Decoder ──► Fused Image
                                                      ▲
Infrared Image ─► Inf Branch (Conv + DRC) ──► Weighted Fusion
                      │                              │
                      └── ScalarPredictor ───────────┘
                           (a1, a2 modality weights)
```

### Key Components

| Module | Description |
|---|---|
| **Dual-branch Encoder** | Separate visible and infrared branches: `ConvLeakyRelu2d` → `DRC` (Dense + Residual + Sobel Edge extraction) |
| **ScalarPredictor** | GAP → FC → Sigmoid, predicts modality importance weights (a1, a2) from concatenated shallow features |
| **SA1Attention** | Spatial attention producing two attention maps (SAM3, SAM4), encouraging cross-modal spatial complementarity |
| **FrozenBranch** | EMA-style frozen copies of encoder branches for stable feature extraction during discriminator training |
| **Dual Discriminators** | `DisVIS_net` and `DisIR_net` operate in feature space (DRC outputs), judging whether features come from source images or fused images |

## Loss Functions

```
L_total = 20·L_intensity + 10·L_gradient + 1·L_GAN + 1·L_SSIM + 0.5·L_attention + 0.1·L_entropy
```

| Loss | Weight | Description |
|---|---|---|
| `L_intensity` | 20 | L1 loss between fused image and element-wise max of visible/infrared Y channels |
| `L_gradient` | 10 | Laplacian gradient consistency with joint source gradients |
| `L_GAN` | 1 | MSE loss for generator against dual discriminators in feature space |
| `L_SSIM` | 1 | Multi-scale SSIM loss against both source modalities |
| `L_attention` | 0.5 | MSE loss encouraging spatial attention maps to be complementary (sum ≈ 1) |
| `L_entropy` | 0.1 | InceptionV3 entropy hinge loss for semantic discrimination enhancement |

## Project Structure

```
C2LNet-main/
├── FusionNet.py              # FusionNet, DRC, DenseBlock, SA1Attention, ScalarPredictor
├── Dual_discriminator.py     # DisVIS_net and DisIR_net (7-layer conv discriminators)
├── loss.py                   # GeneratorLoss, DiscriminatorLoss, InceptionEntropyHingeLoss, SSIM
├── TaskFusion_dataset.py     # MSRS dataset loader (train/val/test splits)
└── train.py                  # Training script + inference (run_fusion)
```

## Dataset

Uses **MSRS** dataset. Expected directory structure:

```
datasets/MSRS/
├── Visible/train/MSRS/       # Visible images (RGB)
├── Infrared/train/MSRS/      # Infrared images (grayscale)
└── Label/train/MSRS/         # Ground truth labels
```

## Training

```bash
python train.py --batch_size 5 --gpu 0 --num_workers 8
```

### Training Details

| Setting | Value |
|---|---|
| Optimizer | Adam |
| Initial LR (Generator) | 1×10⁻³, decay 0.75 per epoch |
| Initial LR (Discriminators) | 5×10⁻⁴, decay 0.75 per epoch |
| Epochs | 20 |
| Batch Size | 5 |
| Color Space | YCrCb (fusion in Y channel, Cb/Cr from visible image) |
| Image Resolution | Variable (no downsampling in fusion module, stride=1, padding=same) |

### Training Process

1. **Discriminator Step**: FrozenBranch encoders extract features from the fused image. `DisIR_net` and `DisVIS_net` classify real (source) vs. fake (fused) features.
2. **Generator Step**: Fuse images, extract features, pass through discriminators. Compute multi-term loss and update FusionNet.
3. **FrozenBranch Sync**: After each epoch, frozen branches are synchronized with the current encoder weights.

## Dependencies

- Python ≥ 3.7
- PyTorch
- torchvision
- NumPy
- OpenCV
- PIL / Pillow
- Matplotlib


