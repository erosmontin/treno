# Treno v3.5.0 Quick Reference Card

## 🚀 Quick Start

```python
from treno import EMUNet, EMUNetMapToMap, EMUNet1D
import torch

# 2D Classification
model = EMUNet(1, 10, dimension=2, task='classification')
x = torch.randn(8, 1, 256, 256)
output = model(x)  # [8, 10]

# 3D Segmentation  
model = EMUNet(1, 5, dimension=3, task='segmentation')
x = torch.randn(4, 1, 128, 128, 128)
mask = model(x)  # [4, 5, 128, 128, 128]

# Map-to-Map (Image Translation)
model = EMUNetMapToMap(1, 1, dimension=2)
noisy = torch.randn(8, 1, 256, 256)
clean = model(noisy)  # [8, 1, 256, 256]

# 1D Time-Series
model = EMUNet1D(1, 5, task='classification')
signal = torch.randn(32, 1, 2048)
output = model(signal)  # [32, 5]
```

---

## 📋 Task Summary

### Classification
```python
EMUNet/EMUNetPP/EMUNet1D(..., task='classification')
# Output: Sigmoid-activated probabilities [B, num_classes]
```

### Regression
```python
EMUNet/EMUNetPP/EMUNet1D(..., task='regression')
# Output: Raw values [B, num_outputs]
```

### Segmentation
```python
EMUNet/EMUNetPP(..., task='segmentation')
# Output: Logits [B, num_classes, ...]
# Use with CrossEntropyLoss
```

### Map-to-Map (NEW)
```python
EMUNetMapToMap/EMUNetPPMapToMap/UNet1DOptimized(...)
# Output: Reconstructed image [B, out_channels, ...]
# Use with L1/L2 reconstruction losses
```

---

## 🎯 Model Selection

### By Dimension

| Input Type | Model |
|-----------|-------|
| 1D Signal (L) | `EMUNet1D` or `UNet1DOptimized` |
| 2D Image (H, W) | `EMUNet` or `EMUNetPP` |
| 3D Volume (D, H, W) | `EMUNet` or `EMUNetPP` |

### By Task

| Task | Models |
|------|--------|
| Classification | `EMUNet`, `EMUNetPP`, `EMUNet1D` |
| Regression | `EMUNet`, `EMUNetPP`, `EMUNet1D` |
| Segmentation | `EMUNet`, `EMUNetPP` |
| Map-to-Map | `EMUNetMapToMap`, `EMUNetPPMapToMap`, `UNet1DOptimized` |

### By Performance

- **Fast:** `EMUNet` (standard U-Net)
- **Better:** `EMUNetPP` (dense skip connections)
- **Specialized 1D:** `UNet1DOptimized` (dilated, optimized for signals)

---

## 🎨 Common Configurations

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
    out_channels=4,
    dimension=3,
    task='segmentation',
    num_filters=[32, 64, 128, 256],
    use_attention=True
)
```

### Image Synthesis (3D)
```python
EMUNetPPMapToMap(
    in_channels=1,  # T2 MRI
    out_channels=1,  # Synthetic T1
    dimension=3,
    num_filters=[64, 128, 256],
    activation_final='sigmoid'
)
```

### ECG Classification (1D)
```python
EMUNet1D(
    in_channels=1,
    out_channels=5,  # Arrhythmia types
    task='classification',
    use_radiomics=True,
    depth=4
)
```

---

## 📊 Loss Functions

### Classification
```python
criterion = nn.BCELoss()  # Multi-label
# or
criterion = nn.CrossEntropyLoss()  # Single-label (segmentation)
```

### Regression
```python
criterion = nn.MSELoss()
# or
criterion = nn.L1Loss()
```

### Map-to-Map
```python
criterion = nn.MSELoss()  # or L1Loss()
# or weighted combination
loss = 0.7 * F.mse_loss(...) + 0.3 * F.l1_loss(...)
```

---

## 🔧 Key Parameters

### Universal Parameters
```python
in_channels        # Input channels (1-4 typical)
out_channels       # Output size
dimension          # 1, 2, or 3
num_filters        # [64, 128, 256, 512] typical
use_attention      # Enable CBAM (default: True)
dropout_rate       # 0.0-0.5 (default: 0.0)
use_batchnorm      # Enable BatchNorm (default: True)
```

### Task-Specific
```python
task              # 'classification', 'regression', 'segmentation', 'map-to-map'
fc_layers         # [1024, 512] for classification/regression
activation        # 'leaky_relu' (default), 'relu', 'gelu', 'none'
```

### Advanced
```python
use_radiomics     # Enable radiomics features
num_bins          # Histogram bins (64-256)
radii             # GLCM radii [1, 2, 3]
extra_params_dim  # Extra features (age, etc.)
```

---

## 💾 Advanced Features

### With Extra Parameters
```python
model = EMUNet(..., extra_params_dim=3)

image = torch.randn(8, 1, 256, 256)
params = torch.randn(8, 3)  # age, sex, weight

output = model(image, params)
```

### With Radiomics
```python
model = EMUNet(..., use_radiomics=True, radii=[1, 2])

# Automatically computes:
# - 21 first-order statistics
# - Multi-directional GLCM
# - All concatenated and normalized
```

### Feature Extraction
```python
features, skip_conns = model.extract_features(x)
print(f"Bottleneck: {features.shape}")
print(f"Skip connections: {[s.shape for s in skip_conns]}")
```

---

## 📈 Performance Tips

### Memory
- Reduce `num_filters` for 3D
- Use smaller batch sizes for 3D (2-4)
- Use `EMUNet` instead of `EMUNetPP` if memory-constrained

### Speed
- Disable radiomics if not needed
- Use `dimension=2` when possible
- Reduce `num_bins` to 64 for radiomics

### Accuracy
- Enable `use_attention=True`
- Use `EMUNetPP` for segmentation
- Increase `dropout_rate` for small datasets
- Use `radii=[1, 2, 3]` for more radiomics features

---

## 🧪 Testing

```bash
# Run all tests
pytest test_nd_comprehensive.py -v

# Run specific tests
pytest test_nd_comprehensive.py::TestMapToMap -v
pytest test_nd_comprehensive.py::TestUNet1D -v
pytest test_nd_comprehensive.py::TestRadiomicsFeatures -v
```

---

## 📚 Documentation

| Document | Content |
|----------|---------|
| `ND_COMPREHENSIVE_GUIDE.md` | Complete reference guide |
| `IMPLEMENTATION_SUMMARY.md` | What was implemented |
| `examples/example_nd_classification.py` | 11 classification examples |
| `examples/example_nd_maptomap.py` | 12 map-to-map examples |
| Inline docstrings | API documentation |

---

## ✅ Compatibility Matrix

| Task | 1D | 2D | 3D | Loss |
|------|----|----|----|----|
| Classification | ✅ | ✅ | ✅ | BCELoss, CrossEntropyLoss |
| Regression | ✅ | ✅ | ✅ | MSELoss, L1Loss |
| Segmentation | ✅ | ✅ | ✅ | CrossEntropyLoss |
| Map-to-Map | ✅ | ✅ | ✅ | MSELoss, L1Loss |

---

## 🚨 Common Issues

### Shape Mismatch
```python
# Wrong
x = torch.randn(8, 256, 256)  # Missing channel dim

# Correct
x = torch.randn(8, 1, 256, 256)  # [B, C, H, W]
```

### Wrong Task Output
```python
# Classification outputs sigmoid-activated values
# Segmentation outputs raw logits
# Map-to-map outputs activated based on activation_final

# Use correct loss:
# classification -> BCELoss
# segmentation -> CrossEntropyLoss
# map-to-map -> L1/L2 Loss
```

### Radiomics Nan Values
```python
# Add epsilon to prevent division by zero
fos = calculate_fos_features(x, num_bins=256)
fos[torch.isnan(fos)] = 0
```

---

## 📞 Getting Help

1. Check `ND_COMPREHENSIVE_GUIDE.md` for detailed examples
2. Run relevant example scripts in `examples/`
3. Review test cases in `test_nd_comprehensive.py`
4. Check inline docstrings: `help(EMUNet)`
5. See GitHub issues and discussions

---

**Treno v3.5.0 - Complete ND Support Ready! 🎉**
