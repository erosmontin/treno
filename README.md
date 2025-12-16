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

### Segmentation (Primary U-Net Use Case)

```python
from treno import EMUNet
import torch

model = EMUNet(
    in_channels=1,           # Single channel (CT/MRI)
    out_channels=4,          # 4 segmentation classes
    dimension=3,             # 3D volumes
    use_radiomics=True,      # Add radiomics features
    use_attention=True,      # Add CBAM attention
    extra_params_dim=3       # Clinical metadata (age, sex, etc.)
)

x = torch.randn(4, 1, 128, 128, 128)
extra = torch.randn(4, 3)  # Clinical data
mask = model(x, extra)  # [4, 4, 128, 128, 128] - segmentation logits
```

### High-Quality Segmentation (U-Net++)

```python
from treno import EMUNetPP  # Dense skip connections for better results

model = EMUNetPP(
    in_channels=1,
    out_channels=4,          # 4 segmentation classes
    dimension=3,
    extra_params_dim=5       # More clinical features
)

x = torch.randn(2, 1, 128, 128, 128)
extra = torch.randn(2, 5)
mask = model(x, extra)  # [2, 4, 128, 128, 128]
```

### Map-to-Map (Image Translation)

```python
from treno import EMUNetMapToMap

model = EMUNetMapToMap(
    in_channels=1,           # Noisy image
    out_channels=1,          # Clean image
    dimension=2,
    extra_params_dim=2,      # Conditioned generation
    use_radiomics=True       # Texture-aware translation
)

noisy = torch.randn(8, 1, 256, 256)
conditions = torch.randn(8, 2)
clean = model(noisy, conditions)  # [8, 1, 256, 256]
```

### Classification (Use LeNet/ResNet)

```python
from treno import EMLeNet  # For classification tasks

model = EMLeNet(
    in_channels=1,
    out_channels=5,          # 5 disease classes
    dimension=3,
    task='classification',
    use_radiomics=True,
    extra_params_dim=3
)

x = torch.randn(4, 1, 128, 128, 128)
extra = torch.randn(4, 3)
probs = model(x, extra)  # [4, 5] - class probabilities
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

### Joint Segmentation + Classification (EMDualHead)

```python
from treno import EMDualHead

model = EMDualHead(
    in_channels=1,
    seg_out_channels=4,      # 4 segmentation classes
    cls_out_channels=3,      # 3 classification classes
    dimension=3,
    cls_task='classification',
    extra_params_dim=5
)

x = torch.randn(2, 1, 64, 64, 64)
extra = torch.randn(2, 5)
seg_out, cls_out = model(x, extra)  # [2, 4, 64, 64, 64], [2, 3]
```

### Autoencoder / VAE (EMAutoEncoder)

```python
from treno import EMAutoEncoder

model = EMAutoEncoder(
    in_channels=1,
    out_channels=1,
    dimension=3,
    latent_dim=128,
    variational=True  # VAE mode
)

x = torch.randn(2, 1, 64, 64, 64)
recon, mu, logvar = model(x)  # Reconstruction + latent params
```

### Deep Radiomics Feature Extraction

```python
from treno import EMUNet, get_deep_radiomics_features
import torch

# After training a model for its task, extract deep radiomics vectors
model = EMUNet(in_channels=1, out_channels=4, dimension=3, use_radiomics=True)
model.eval()

x = torch.randn(2, 1, 64, 64, 64)

# Returns [B, F] feature vectors (pooled deep features + engineered radiomics)
feat = get_deep_radiomics_features(model, x, pool='avg', include_engineered=True)
print(feat.shape)  # e.g., [2, 64 + R]
```

---

## 📊 Task Compatibility Matrix

| Task | 1D | 2D | 3D | Recommended Models |
|------|----|----|----|----|
| **Segmentation** | ✅ | ✅ | ✅ | `EMUNet`, `EMUNetPP` |
| **Image Translation** | ✅ | ✅ | ✅ | `EMUNetMapToMap`, `EMUNetPPMapToMap` |
| **Classification** | ✅ | ✅ | ✅ | `EMLeNet`, `EMResNet` |
| **Regression** | ✅ | ✅ | ✅ | `EMLeNet`, `EMResNet` |
| **Joint Seg+Cls** | ✅ | ✅ | ✅ | `EMDualHead` |
| **Autoencoding** | ✅ | ✅ | ✅ | `EMAutoEncoder` |

---

## 🏗️ Model Selection Guide

### Segmentation Models
| Model | Description | When to Use |
|-------|-------------|-------------|
| `EMUNet` | Standard U-Net with skip connections | Fast baseline, general purpose segmentation |
| `EMUNetPP` | U-Net++ with dense nested skips | Better accuracy, more parameters |

### Image Translation Models
| Model | Description | When to Use |
|-------|-------------|-------------|
| `EMUNetMapToMap` | U-Net for map-to-map tasks | Denoising, synthesis, domain transfer |
| `EMUNetPPMapToMap` | U-Net++ for high-quality reconstruction | Superior image quality, style transfer |

### Classification/Regression Models
| Model | Description | When to Use |
|-------|-------------|-------------|
| `EMLeNet` | Lightweight CNN | Fast inference, smaller datasets |
| `EMResNet` | Deep residual encoder | Complex features, large datasets |

### Specialized Models
| Model | Description | When to Use |
|-------|-------------|-------------|
| `EMDualHead` | Joint segmentation + classification | Multi-task learning, shared features |
| `EMAutoEncoder` | VAE/AE for latent space | Anomaly detection, feature learning |

**All models support:**
- ✅ 1D/2D/3D inputs
- ✅ Radiomics features
- ✅ Extra parameters (clinical metadata)
- ✅ Fusion gating (when extra_params_dim > 0)
- ✅ CBAM attention mechanisms

---

## 🏋️ Trainer Class

Built-in training utilities with TensorBoard support:

```python
from treno import Trainer
from torch.utils.tensorboard import SummaryWriter

trainer = Trainer(
    model=model,
    optimizer=optimizer,
    task='segmentation',      # 'classification', 'regression', 'segmentation', 'dual'
    loss_fn=loss_fn,
    scheduler=scheduler,
    device='cuda',
    use_amp=True,             # Mixed precision
    gradient_clip=1.0,
    early_stopping_patience=10
)

# Train with TensorBoard logging
writer = SummaryWriter('runs/experiment')
history = trainer.fit(
    train_loader,
    val_loader,
    epochs=100,
    writer=writer             # Auto-logs losses and metrics
)

# Evaluate
results = trainer.evaluate(test_loader, return_predictions=True)
```

### Custom TensorBoard Logging

```python
class MyTrainer(Trainer):
    def on_epoch_end(self, phase, epoch, loss, metrics):
        super().on_epoch_end(phase, epoch, loss, metrics)  # Default logging
        if self.writer and phase == 'train':
            lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('LR', lr, epoch)
    
    def on_batch_end(self, phase, batch_idx, loss, output, targets):
        # Custom per-batch logging
        pass
    
    def on_test_end(self, results):
        # Custom test completion logging
        pass
```

---

## 📚 Documentation

- **README.md** (this file) - Overview and quick start
- **LLM_AGENT_GUIDE.md** - Comprehensive guide for LLM agents building pipelines (includes explainability, migration, API reference)
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

```

---

## Refactoring Highlights (v3.5)

- U-Net family now strictly segmentation-only (clear task separation)
- Map-to-Map models support conditioned generation via fusion
- Deep radiomics unified: pooled deep features + engineered radiomics
- 1D/2D/3D parity across models with consistent APIs
- Documentation consolidated into this README and LLM guide

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
