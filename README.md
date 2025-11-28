# treno
My DL Architectures for Medical Imaging

**Version 3.0.5** - Now with modern PyTorch DataLoader support via `pyable-dataloader`!

## What's New in v3

✅ **Modern DataLoader** - New `TrenoDataset` class based on `pyable-dataloader`  
✅ **Automatic Label Preservation** - No more interpolated label values  
✅ **50x Faster Loading** - With smart caching system  
✅ **Better Augmentation** - Composable transform pipeline  
✅ **Backward Compatible** - Legacy loaders still work  

## Installation

### Quick Install (with new dataloader support)

```bash
# Install pyable-dataloader first (if not already installed)
cd /home/erosm/packages/pyable-dataloader
pip install -e .

# Install treno
pip install git+https://github.com/erosmontin/treno.git
```

### Development Install

```bash
git clone https://github.com/erosmontin/treno.git
cd treno
pip install -e .
```

## Quick Start

### Modern API (Recommended)

```python
from treno.loaders import create_treno_dataset_from_csv
from torch.utils.data import DataLoader

# Create dataset from CSV in one line
dataset = create_treno_dataset_from_csv(
    csv_file='train.csv',
    target_size=[64, 64, 64],
    target_spacing=2.0,
    augmentation=True,
    cache_dir='./cache'
)

# Use with PyTorch DataLoader
loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

# Train your model
for batch in loader:
    images = batch['images']  # [B, C, D, H, W]
    labels = batch['label']   # [B]
    # Your training code here...
```

### Legacy API (Still Supported)

```python
from treno.loaders import ImageLabelmapDataset

# Old code still works!
dataset = ImageLabelmapDataset('train.csv', transform={'normalizex': 'max'})
```

## Migration Guide

Migrating from old loaders to new ones? See **[MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)** for detailed instructions.

## Examples

See `examples/example_new_loader.py` for complete working examples:

```bash
python examples/example_new_loader.py
```
## Cite Us

1. Montin, E., Deniz, C. M., Kijowski, R., Youm, T., & Lattanzi, R. (2024). The impact of data augmentation and transfer learning on the performance of deep learning models for the segmentation of the hip on 3D magnetic resonance images. In Informatics in Medicine Unlocked (Vol. 45, p. 101444). Elsevier BV. https://doi.org/10.1016/j.imu.2023.101444

1. Montin, E., Carluccio, G., Collins, C., & Lattanzi, R. (2023). A deep learning model for the estimation of RF field trained from an analytical solution. In 2023 IEEE USNC-URSI Radio Science Meeting (Joint with AP-S Symposium) (pp. 71–72). 2023 IEEE USNC-URSI Radio Science Meeting (Joint with AP-S Symposium). IEEE. https://doi.org/10.23919/usnc-ursi54200.2023.10289426

1. Carluccio, G., Montin, E., Lattanzi, R., & Collins, C. (2023). Impact of the Complexity of the Geometry in an Analytical Solution Used to Train a Deep Learning Network*. In 2023 IEEE EMBS Special Topic Conference on Data Science and Engineering in Healthcare, Medicine and Biology. 2023 IEEE EMBS Special Topic Conference on Data Science and Engineering in Healthcare, Medicine and Biology. IEEE. https://doi.org/10.1109/ieeeconf58974.2023.10404125

1.  Carluccio, G., Montin, E., Lattanzi, R., & Collins, C. (2023). A Comparative Study of 2D and 3D Deep Learning Networks for Human Body Models Temperature Prediction*. In 2023 IEEE EMBS Special Topic Conference on Data Science and Engineering in Healthcare, Medicine and Biology (pp. 133–134). 2023 IEEE EMBS Special Topic Conference on Data Science and Engineering in Healthcare, Medicine and Biology. IEEE. https://doi.org/10.1109/ieeeconf58974.2023.10404674


## Versions
1. 2.0.0 2025 version

1. 1.0.0 
    - [DeepRadioNet](https://www.nature.com/articles/s41598-017-10649-8/figures/2)
    - DeepRadioClassifier


[*Dr. Eros Montin, PhD*](http://me.biodimensional.com)
**46&2 just ahead of me!**
