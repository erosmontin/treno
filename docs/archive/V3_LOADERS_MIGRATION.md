# Treno v3 Loaders Migration Guide

## Summary of Changes

Treno v3 has been completely modernized to use `pyable-dataloader` for all data loading functionality. All legacy loader classes have been **removed**.

## ✅ What's New

### Modern Loader Classes
- `TrenoDataset`: Unified dataset class wrapping PyableDataset
- `create_manifest_from_csv()`: Convert CSV to manifest format
- `create_treno_dataset_from_csv()`: Convenience function for quick setup
- `save_manifest()` / `load_manifest()`: Manifest I/O utilities

### Key Benefits
- Uses modern `PyableDataset` from `pyable-dataloader`
- Consistent API across all packages
- Better caching support
- Modern transforms (IntensityNormalization, RandomFlip, etc.)
- Cleaner, more maintainable code

## ❌ What's Removed

The following legacy classes are **no longer available** in treno v3:
- `ImageImageDataset`
- `ImageLabelmapDataset`
- `ImaImaDataset`
- `ImaRoiDataset`
- `normalize()` function
- `labelMapToChannel()` function
- `possibletransforms()` function
- `ImaginableDataloader()` function

## 🔄 Migration Examples

### Before (Legacy)
```python
from treno.loaders import ImageImageDataset

dataset = ImageImageDataset(
    annotations_file='train.csv',
    transform={'normalizex': 'z', 'resize': [64, 64, 64]}
)
```

### After (v3)
```python
from treno import create_treno_dataset_from_csv
from pyable_dataloader import Compose, IntensityNormalization

dataset = create_treno_dataset_from_csv(
    csv_file='train.csv',
    target_size=[64, 64, 64],
    augmentation=True  # Applies zscore normalization + random flip
)
```

### Custom Transforms (v3)
```python
from treno import TrenoDataset, create_manifest_from_csv
from pyable_dataloader import Compose, IntensityNormalization, RandomFlip

# Create manifest
manifest = create_manifest_from_csv('train.csv')

# Define transforms
transforms = Compose([
    IntensityNormalization(method='zscore'),
    RandomFlip(axes=[1, 2], prob=0.5)
])

# Create dataset
dataset = TrenoDataset(
    manifest=manifest,
    target_size=[64, 64, 64],
    target_spacing=2.0,
    transforms=transforms,
    cache_dir='./cache'
)
```

## 📦 Package Dependencies

Treno v3 requires:
- `pyable-dataloader` (mandatory)
- `pyable` v3.x
- `pynico` (for utilities)
- PyTorch
- SimpleITK (via pyable)

## 🔗 Related Changes

### Package Naming (v3)
- ✅ `pynico` (was: `pynico_eros_montin`)
- ✅ `pyable` (was: `pyable_eros_montin`)
- ✅ All imports updated throughout codebase

### Utils Module
- Removed dependency on `pynico.stats` (removed in pynico v3)
- Added local implementations of confusion matrix helpers

## 📝 CSV Format

The CSV format remains compatible. Example:

```csv
label,image1,image2,roi
0,/path/to/img1.nii.gz,/path/to/img2.nii.gz,/path/to/roi.nii.gz
1,/path/to/img3.nii.gz,/path/to/img4.nii.gz,/path/to/roi2.nii.gz
```

This gets automatically converted to manifest format by `create_manifest_from_csv()`.

## 🚀 Getting Started

```bash
# Install all packages
pip install -e /path/to/pynico
pip install -e /path/to/pyable
pip install -e /path/to/pyable-ml
pip install -e /path/to/pyable-dataloader
pip install -e /path/to/pyfe
pip install -e /path/to/treno
```

```python
# Quick start
from treno import create_treno_dataset_from_csv
from torch.utils.data import DataLoader

dataset = create_treno_dataset_from_csv(
    'train.csv',
    target_size=[64, 64, 64],
    target_spacing=2.0,
    augmentation=True
)

loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

for batch in loader:
    images = batch['images']  # [B, C, D, H, W]
    labels = batch['label']   # [B]
    # Train model...
```

## ⚠️ Breaking Changes

1. **All legacy loaders removed** - Must migrate to TrenoDataset/PyableDataset
2. **Different return format** - Returns dict with keys: `'images'`, `'rois'`, `'labelmaps'`, `'label'`, `'id'`
3. **Transform API changed** - Use Compose with pyable-dataloader transforms
4. **Import paths** - Import from `pynico`, `pyable` (not `*_eros_montin`)

## 📚 Documentation

- TrenoDataset: See docstring in `treno/loaders.py`
- PyableDataset: See `pyable-dataloader/README.md`
- Transforms: See `pyable-dataloader/ARCHITECTURE.md`

## 🐛 Troubleshooting

**ImportError: cannot import name 'ImageImageDataset'**
- Solution: This class is removed in v3. Use `TrenoDataset` instead.

**Package name mismatch errors**
- Solution: Use `pynico` and `pyable` (not `*_eros_montin` versions)

**Missing pyable-dataloader**
- Solution: `pip install -e /path/to/pyable-dataloader`

## 📞 Support

For issues or questions about the v3 migration, please refer to:
- `treno/README.md`
- `pyable-dataloader/QUICKSTART.md`
- Package documentation in each repository
