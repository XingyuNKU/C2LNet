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
python train.py 
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


