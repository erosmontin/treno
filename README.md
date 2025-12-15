# Treno

**Deep Learning Architectures for Medical Imaging**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Version 3.5.0** - Complete n-dimensional (1D/2D/3D) support for all task types!

---

## 🚀 Features

### Core Capabilities

✅ **All Dimensions**: 1D (signals/time-series), 2D (images), 3D (volumes)  
✅ **All Tasks**: Classification, Regression, Segmentation, Map-to-Map (image translation)  
✅ **Modern Data Loaders**: Based on `pyable-dataloader` with smart caching  
✅ **Radiomics Integration**: Automated first-order statistics + multi-directional GLCM  
✅ **Attention Mechanisms**: CBAM (Convolutional Block Attention Module)  
✅ **Extra Parameters**: Inject clinical metadata (age, sex, imaging parameters, etc.)  
✅ **Production Ready**: Fully tested with 51 comprehensive test cases  

### What's New in v3.5

🎯 **Map-to-Map Models** - Image-to-image translation tasks  
🎯 **1D-Optimized U-Net** - Dilated convolutions for time-series  
🎯 **Multi-directional GLCM** - True ND radiomics across all spatial axes  
🎯 **Skip Connection Aligner** - Handle spatial mismatches flexibly  

---

## 📦 Installation

### Quick Install

```bash
# Install pyable-dataloader first (if not already installed)
pip install pyable-dataloader

# Install treno
pip install git+https://github.com/erosmontin/treno.git
```

### Development Install

```bash
git clone https://github.com/erosmontin/treno.git
cd treno
pip install -e .
```

---

## 🎯 Quick Start

### Classification (3D Medical Images)

```python
from treno import EMUNet
import torch

model = EMUNet(
    in_channels=1,           # Single channel (CT/MRI)
    out_channels=5,          # 5 disease classes
    dimension=3,             # 3D volumes
    task='classification',
    use_radiomics=True,      # Add radiomics features
    use_attention=True       # Add CBAM attention
)

x = torch.randn(4, 1, 128, 128, 128)
output = model(x)  # [4, 5] - class probabilities
```

### Segmentation

```python
from treno import EMUNetPP  # U-Net++ for better results

model = EMUNetPP(
    in_channels=1,
    out_channels=4,          # 4 segmentation classes
    dimension=3,
    task='segmentation'
)

x = torch.randn(2, 1, 128, 128, 128)
mask = model(x)  # [2, 4, 128, 128, 128]
```

### Map-to-Map (Image Translation)

```python
from treno import EMUNetMapToMap

model = EMUNetMapToMap(
    in_channels=1,           # Noisy image
    out_channels=1,          # Clean image
    dimension=2
)

noisy = torch.randn(8, 1, 256, 256)
clean = model(noisy)  # [8, 1, 256, 256]
```

### 1D Time-Series

```python
from treno import EMUNet1D

model = EMUNet1D(
    in_channels=1,
    out_channels=3,
    task='classification'
)

signal = torch.randn(32, 1, 2048)
output = model(signal)  # [32, 3]
```

### Data Loading

```python
from treno import create_treno_dataset_from_csv
from torch.utils.data import DataLoader

dataset = create_treno_dataset_from_csv(
    csv_file='train.csv',
    target_size=[64, 64, 64],
    target_spacing=2.0,
    augmentation=True,
    cache_dir='./cache'
)

loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

for batch in loader:
    images = batch['images']  # [B, C, D, H, W]
    labels = batch['label']   # [B]
```

---

## 📊 Task Compatibility Matrix

| Task | 1D | 2D | 3D | Models |
|------|----|----|----|----|
| **Classification** | ✅ | ✅ | ✅ | `EMUNet`, `EMUNetPP`, `EMUNet1D` |
| **Regression** | ✅ | ✅ | ✅ | `EMUNet`, `EMUNetPP`, `EMUNet1D` |
| **Segmentation** | ✅ | ✅ | ✅ | `EMUNet`, `EMUNetPP` |
| **Map-to-Map** | ✅ | ✅ | ✅ | `EMUNetMapToMap`, `EMUNetPPMapToMap`, `UNet1DOptimized` |

---

## 🏗️ Model Selection Guide

| Model | Description | When to Use |
|-------|-------------|-------------|
| `EMUNet` | Standard U-Net | Fast baseline, general purpose |
| `EMUNetPP` | U-Net++ (dense skip connections) | Better accuracy, more parameters |
| `EMUNet1D` | Full-featured 1D U-Net | Time-series with radiomics |
| `UNet1DOptimized` | Dilated 1D U-Net | Variable-length sequences |
| `EMUNetMapToMap` | U-Net for image translation | Denoising, synthesis, style transfer |
| `EMUNetPPMapToMap` | U-Net++ for reconstruction | Superior image quality |

---

## 📚 Documentation

- **README.md** (this file) - Overview and quick start
- **LLM_AGENT_GUIDE.md** - Comprehensive guide for LLM agents building pipelines
- **GRADCAM_GUIDE.md** - Explainability and visualization
- **MIGRATION_GUIDE.md** - Migrating from legacy data loaders
- **examples/** - Working example scripts

---

## 🧪 Examples

```bash
# Classification examples
python examples/example_nd_classification.py

# Map-to-map examples
python examples/example_nd_maptomap.py

# Data loader examples
python examples/example_new_loader.py

# Grad-CAM visualization
python examples/example_gradcam_saliency.py
```

---

## 🎓 Citation

1. Montin, E., Deniz, C. M., Kijowski, R., Youm, T., & Lattanzi, R. (2024). The impact of data augmentation and transfer learning on the performance of deep learning models for the segmentation of the hip on 3D magnetic resonance images. *Informatics in Medicine Unlocked*, 45, 101444. https://doi.org/10.1016/j.imu.2023.101444

2. Montin, E., Carluccio, G., Collins, C., & Lattanzi, R. (2023). A deep learning model for the estimation of RF field trained from an analytical solution. In *2023 IEEE USNC-URSI Radio Science Meeting* (pp. 71–72). IEEE. https://doi.org/10.23919/usnc-ursi54200.2023.10289426

3. Carluccio, G., Montin, E., Lattanzi, R., & Collins, C. (2023). Impact of the Complexity of the Geometry in an Analytical Solution Used to Train a Deep Learning Network. In *2023 IEEE EMBS Special Topic Conference*. IEEE. https://doi.org/10.1109/ieeeconf58974.2023.10404125

4. Carluccio, G., Montin, E., Lattanzi, R., & Collins, C. (2023). A Comparative Study of 2D and 3D Deep Learning Networks for Human Body Models Temperature Prediction. In *2023 IEEE EMBS Special Topic Conference* (pp. 133–134). IEEE. https://doi.org/10.1109/ieeeconf58974.2023.10404674

---

## 🔄 Version History

### v3.5.0 (December 2025)
- ✨ Complete ND support (1D/2D/3D) across all tasks
- ✨ Map-to-map models for image translation
- ✨ 1D-optimized U-Net with dilated convolutions
- ✨ Multi-directional GLCM radiomics
- ✨ 51 comprehensive test cases

### v3.0.5 (2025)
- ✨ Modern PyTorch DataLoader via `pyable-dataloader`
- ✨ 50x faster loading with smart caching
- ✅ Backward compatible

### v2.0.0 / v1.0.0
- DeepRadioNet, DeepRadioClassifier

---

## 👤 Author

**Dr. Eros Montin, PhD**  
📧 eros.montin@gmail.com  
🔗 [GitHub](https://github.com/erosmontin/treno)

---

## 📝 License

MIT License - see [LICENSE](LICENSE) file

---

**Ready to build medical imaging AI? Start with `LLM_AGENT_GUIDE.md`! 🚀**


[*Dr. Eros Montin, PhD*](http://me.biodimensional.com)
**46&2 just ahead of me!**
