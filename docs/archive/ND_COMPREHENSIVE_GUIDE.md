# ND Comprehensive Guide: All Dimensions, All Tasks

This guide documents the full ND (1D, 2D, 3D) support across all task types in Treno v3.5+.

## Quick Reference: Task Compatibility Matrix

| Task | 1D | 2D | 3D | Status | Class |
|------|----|----|----|---------|----|
| **Classification** | ✅ | ✅ | ✅ | READY | `EMUNet`, `EMUNetPP`, `EMUNet1D` |
| **Regression** | ✅ | ✅ | ✅ | READY | `EMUNet`, `EMUNetPP`, `EMUNet1D` |
| **Segmentation** | ✅ | ✅ | ✅ | READY | `EMUNet`, `EMUNetPP` |
| **Map-to-Map** | ✅ | ✅ | ✅ | **NEW** | `EMUNetMapToMap`, `EMUNetPPMapToMap`, `UNet1DOptimized` |

## Features Overview

### ✨ New in v3.5

1. **Map-to-Map Models** - Image-to-image translation across all dimensions
   - `MapToMapHead`: Dense prediction head for image reconstruction
   - `EMUNetMapToMap`: Standard U-Net with map-to-map head
   - `EMUNetPPMapToMap`: U-Net++ with dense skip connections

2. **1D-Optimized U-Net** - Specialized for time-series and signals
   - `UNet1DOptimized`: Dilated convolutions, temporal pooling
   - `EMUNet1D`: Full-featured with radiomics and extra parameters
   - Optimized for variable-length sequences

3. **Improved Radiomics** - True multi-directional GLCM
   - Now computes GLCM along all spatial axes
   - 1D: 1 axis direction
   - 2D: 2 axis directions (vertical, horizontal)
   - 3D: 3 axis directions (depth, height, width)

4. **Skip Connection Alignment** - Handle spatial mismatches
   - `SkipConnectionAligner`: Flexible padding strategies
   - Supports 'pad', 'crop', 'interpolate' modes

---

## Classification

### 2D Classification (Basic)

```python
from treno.models import EMUNet
import torch

# Create model
model = EMUNet(
    in_channels=3,           # RGB image
    out_channels=10,         # 10 classes
    dimension=2,
    task='classification'
)

# Forward pass
x = torch.randn(8, 3, 256, 256)  # [batch, channels, height, width]
logits = model(x)                 # [8, 10] - probabilities
```

### 3D Classification with Radiomics

```python
model = EMUNet(
    in_channels=1,           # Single channel (CT/MRI)
    out_channels=4,          # 4 classes
    dimension=3,
    task='classification',
    use_radiomics=True,      # Compute radiomics features
    num_bins=64,             # Histogram bins
)

x = torch.randn(4, 1, 128, 128, 128)  # 3D volume
output = model(x)  # [4, 4]
```

### 1D Classification (Time-Series)

```python
from treno.unet_1d_optimized import EMUNet1D

model = EMUNet1D(
    in_channels=1,            # Univariate time-series
    out_channels=3,           # 3 classes
    task='classification'
)

x = torch.randn(32, 1, 512)  # [batch, channels, sequence_length]
output = model(x)             # [32, 3]
```

### Classification with Extra Parameters (Age, TR, TE, etc.)

```python
model = EMUNet(
    in_channels=1,
    out_channels=5,
    dimension=2,
    task='classification',
    extra_params_dim=3  # age, sex, weight
)

image = torch.randn(8, 1, 256, 256)
params = torch.randn(8, 3)  # Extra features: [age, sex, weight]

output = model(image, params)  # [8, 5]
```

---

## Regression

### 2D Regression (Image Quality Assessment)

```python
model = EMUNet(
    in_channels=1,
    out_channels=1,      # Single continuous output (0-100 score)
    dimension=2,
    task='regression'
)

x = torch.randn(16, 1, 512, 512)
quality_score = model(x)  # [16, 1]
```

### 3D Regression with Extra Parameters

```python
model = EMUNet(
    in_channels=2,           # Multi-modal (T1, T2)
    out_channels=1,          # Predict age from brain MRI
    dimension=3,
    task='regression',
    extra_params_dim=2,      # brain_volume, ICV
    use_radiomics=True
)

x = torch.randn(8, 2, 64, 64, 64)
extra = torch.randn(8, 2)  # Brain metrics

predicted_age = model(x, extra)  # [8, 1]
```

### 1D Regression (Signal Processing)

```python
model = EMUNet1D(
    in_channels=1,
    out_channels=1,
    task='regression'
)

# ECG signal -> predict heart rate
signal = torch.randn(32, 1, 1000)  # 1000 samples per signal
heart_rate = model(signal)          # [32, 1]
```

---

## Segmentation

### 2D Segmentation (Medical Image Segmentation)

```python
model = EMUNet(
    in_channels=1,
    out_channels=5,        # Background + 4 organs
    dimension=2,
    task='segmentation',
    use_attention=True     # CBAM attention blocks
)

x = torch.randn(4, 1, 512, 512)
segmentation_logits = model(x)  # [4, 5, 512, 512] - use with CrossEntropyLoss
```

### 3D Segmentation with Dense Skip Connections

```python
model = EMUNetPP(
    in_channels=1,
    out_channels=3,      # 3 tissue classes
    dimension=3,
    task='segmentation',
    num_filters=[32, 64, 128]  # Smaller for memory
)

x = torch.randn(2, 1, 128, 128, 128)
mask = model(x)  # [2, 3, 128, 128, 128]
```

### Loss Functions for Segmentation

```python
import torch.nn.functional as F

# CrossEntropyLoss for multi-class segmentation
criterion = nn.CrossEntropyLoss()
loss = criterion(logits, target)

# Dice loss for imbalanced data
from losses import dice_coefficient

dice = dice_coefficient(hist)
```

---

## Map-to-Map (Image-to-Image Translation)

### 2D Image Denoising

```python
from treno.models import EMUNetMapToMap

model = EMUNetMapToMap(
    in_channels=1,
    out_channels=1,
    dimension=2,
    activation_final='none'  # No activation - raw reconstruction
)

# Noisy image -> Clean image
noisy = torch.randn(8, 1, 256, 256)
clean = model(noisy)  # [8, 1, 256, 256]

# Loss: L1 or L2 reconstruction
mse_loss = F.mse_loss(clean, target)
```

### 3D Medical Image Synthesis

```python
from treno.models import EMUNetPPMapToMap

model = EMUNetPPMapToMap(
    in_channels=1,         # T2 MRI
    out_channels=1,        # Synthetic T1 MRI
    dimension=3,
    activation_final='sigmoid'  # Normalize to [0, 1]
)

t2_image = torch.randn(4, 1, 128, 128, 128)
synthetic_t1 = model(t2_image)

# Loss
l1_loss = F.l1_loss(synthetic_t1, real_t1)
```

### 1D Signal Enhancement

```python
from treno.unet_1d_optimized import UNet1DOptimized

model = UNet1DOptimized(
    in_channels=2,         # Noisy ECG + reference signal
    out_channels=1,        # Cleaned ECG
    depth=4,
    task='map-to-map'
)

noisy_ecg = torch.randn(32, 2, 2048)
clean_ecg = model(noisy_ecg)  # [32, 1, 2048]
```

### Multi-Channel Image Translation

```python
model = EMUNetMapToMap(
    in_channels=3,         # RGB image
    out_channels=3,        # RGB artifact-free
    dimension=2,
    activation_final='sigmoid'
)

corrupted = torch.randn(8, 3, 512, 512)
restored = model(corrupted)  # [8, 3, 512, 512]
```

---

## Advanced Features

### Radiomics in All Dimensions

The improved radiomics computation now includes true multi-directional GLCM:

```python
# 1D: 1 direction (temporal)
model_1d = EMUNet1D(..., use_radiomics=True)
# Features: 21 FOS + 3 GLCM = 24 per channel

# 2D: 2 directions (vertical, horizontal)
model_2d = EMUNet(..., dimension=2, use_radiomics=True)
# Features: 21 FOS + 6 GLCM = 27 per channel

# 3D: 3 directions (Z, Y, X)
model_3d = EMUNet(..., dimension=3, use_radiomics=True)
# Features: 21 FOS + 9 GLCM = 30 per channel
```

### Skip Connection Alignment for Flexible Architectures

```python
from treno.models import SkipConnectionAligner

aligner = SkipConnectionAligner(strategy='interpolate')

# Use in custom models
encoder_features = torch.randn(4, 128, 64, 64)
decoder_features = torch.randn(4, 128, 32, 32)

aligned = aligner(encoder_features, decoder_features, dimension=2)
# Output: [4, 128, 64, 64]
```

---

## Feature Extraction & Visualization

### Extract Bottleneck Features

```python
model = EMUNet(..., task='segmentation')

# Forward pass
x = torch.randn(4, 1, 256, 256)
output = model(x)

# Extract features
features, skip_connections = model.extract_features(x)
print(f"Bottleneck shape: {features.shape}")
```

### Radiomics Feature Analysis

```python
from treno.models import calculate_fos_features, calculate_simple_glcm_features

# For custom radiomics analysis
image = torch.randn(128, 128, 128)

# First-order statistics
fos = calculate_fos_features(image, num_bins=256)
print(f"FOS features: {fos.shape[0]} features")

# GLCM features with multiple radii
glcm = calculate_simple_glcm_features(image, radii=[1, 2, 3], dimension=3)
print(f"GLCM features: {glcm.shape[0]} features")
```

---

## Recommended Configurations by Task

### Medical Image Classification (3D)

```python
EMUNet(
    in_channels=1,
    out_channels=num_classes,
    dimension=3,
    task='classification',
    num_filters=[64, 128, 256, 512],
    use_attention=True,
    use_radiomics=True,
    dropout_rate=0.3
)
```

### Multi-Modal Segmentation (3D)

```python
EMUNetPP(
    in_channels=4,  # T1, T2, FLAIR, ADC
    out_channels=4,  # 4 tissue classes
    dimension=3,
    task='segmentation',
    num_filters=[32, 64, 128, 256],
    use_attention=True,
    dropout_rate=0.2
)
```

### Image Synthesis (3D)

```python
EMUNetPPMapToMap(
    in_channels=1,
    out_channels=1,
    dimension=3,
    num_filters=[64, 128, 256],
    activation_final='sigmoid'
)
```

### Time-Series Analysis (1D)

```python
EMUNet1D(
    in_channels=num_signals,
    out_channels=num_classes,
    task='classification',
    depth=4,
    base_filters=32,
    use_radiomics=True
)
```

---

## Testing

Run comprehensive tests for all dimensions and tasks:

```bash
# Run all tests
pytest test_nd_comprehensive.py -v

# Run specific test class
pytest test_nd_comprehensive.py::TestMapToMap -v

# Run specific dimension tests
pytest test_nd_comprehensive.py::TestDimensionSupport -v
```

---

## Performance Tips

1. **Memory Usage**:
   - 1D: Very lightweight, use larger batch sizes
   - 2D: Standard, 8-16 batch typical
   - 3D: Memory intensive, consider 2-4 batch

2. **Radiomics Computation**:
   - Use `num_bins=64` for speed, `256` for accuracy
   - Set `radii=[1]` for speed, `[1, 2, 3]` for detail

3. **Regularization**:
   - Increase `dropout_rate` for small datasets (0.3-0.5)
   - Enable `use_attention=True` for improved learning

4. **1D Signals**:
   - Use `depth=3-4` for reasonable receptive field
   - Dilated convolutions help with long sequences
   - Temporal pooling helps with variable lengths

---

## Migration from v3.0

If upgrading from v3.0, existing code continues to work:

```python
# Old code still works!
model = EMUNet(in_channels=1, out_channels=10, dimension=2, task='classification')

# New: Add map-to-map support
model_new = EMUNetMapToMap(in_channels=1, out_channels=1, dimension=2)

# New: 1D support for time-series
model_1d = EMUNet1D(in_channels=1, out_channels=3, task='classification')
```

---

## Examples

See the `examples/` directory for complete working examples:

- `example_nd_classification.py` - Classification across 1D/2D/3D
- `example_nd_regression.py` - Regression tasks
- `example_nd_segmentation.py` - Segmentation workflows
- `example_nd_maptomap.py` - Image translation and synthesis

---

## Troubleshooting

### Shape Mismatch Errors

```python
# Error: Expected 3D input [B, C, L], got [B, L]
# Fix: Reshape signal properly
signal = signal.unsqueeze(1)  # Add channel dimension
```

### Device Issues

```python
# Ensure all tensors are on same device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
x = x.to(device)
```

### Radiomics Nan Values

```python
# Add epsilon to avoid division by zero
fos = calculate_fos_features(x, num_bins=256)
fos[torch.isnan(fos)] = 0
```

---

## Citation

If you use Treno's ND functionality, please cite:

```bibtex
@software{treno2025,
  title = {Treno: ND Deep Learning Architectures},
  author = {Montin, Eros},
  year = {2025},
  url = {https://github.com/erosmontin/treno}
}
```
