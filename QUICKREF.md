# Treno Quick Reference

## Installation

```bash
pip install git+https://github.com/erosmontin/treno.git
```

## Import

```python
# Modern API
from treno.loaders import TrenoDataset, create_treno_dataset_from_csv

# Transforms (from pyable-dataloader)
from pyable_dataloader import Compose, IntensityNormalization, RandomFlip

# Legacy API
from treno.loaders import ImageImageDataset, ImageLabelmapDataset
```

## Quick Start (Modern API)

### One-Liner from CSV

```python
from treno.loaders import create_treno_dataset_from_csv
from torch.utils.data import DataLoader

dataset = create_treno_dataset_from_csv(
    'train.csv',
    target_size=[64, 64, 64],
    target_spacing=2.0,
    augmentation=True
)

loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)
```

### With Custom Transforms

```python
from treno.loaders import TrenoDataset, create_manifest_from_csv
from pyable_dataloader import Compose, IntensityNormalization, RandomFlip, RandomRotation90

# Convert CSV to manifest
manifest = create_manifest_from_csv(
    'train.csv',
    image_columns=['T1', 'T2'],
    labelmap_column='segmentation',
    label_column='label'
)

# Define transforms
transforms = Compose([
    IntensityNormalization(method='zscore'),
    RandomFlip(axes=[1, 2], prob=0.5),
    RandomRotation90(axes=(1, 2), prob=0.3)
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

## Common Use Cases

### Classification Task

```python
dataset = create_treno_dataset_from_csv(
    'train.csv',
    target_size=[64, 64, 64],
    target_spacing=2.0,
    augmentation=True
)

for batch in loader:
    images = batch['images']  # [B, C, D, H, W]
    labels = batch['label']   # [B]
    
    outputs = model(images)
    loss = criterion(outputs, labels)
```

### Segmentation Task

```python
manifest = create_manifest_from_csv(
    'train.csv',
    image_columns=['image'],
    labelmap_column='segmentation'
)

dataset = TrenoDataset(
    manifest=manifest,
    target_size=[64, 64, 64],
    transforms=transforms
)

for batch in loader:
    images = batch['images']      # [B, C, D, H, W]
    labelmaps = batch['labelmaps']  # [B, 1, D, H, W]
    
    outputs = model(images)
    loss = criterion(outputs, labelmaps.squeeze(1).long())
```

### Multi-Modal Input

```python
manifest = create_manifest_from_csv(
    'train.csv',
    image_columns=['T1', 'T2', 'FLAIR'],
    labelmap_column='segmentation'
)

dataset = TrenoDataset(
    manifest=manifest,
    target_size=[64, 64, 64],
    stack_channels=True,  # Stack as channels
    transforms=transforms
)

# Output: images shape = [B, 3, D, H, W] (3 channels for T1, T2, FLAIR)
```

### ROI-Based Training

```python
manifest = create_manifest_from_csv(
    'train.csv',
    image_columns=['image'],
    roi_column='roi',
    label_column='label'
)

dataset = TrenoDataset(
    manifest=manifest,
    target_size=[64, 64, 64],
    roi_mask=True,      # Mask image with ROI
    roi_center=True,    # Center volume around ROI
    roi_dilation=5.0    # Dilate ROI by 5mm
)
```

## Available Transforms

```python
from pyable_dataloader import (
    Compose,
    IntensityNormalization,  # method='zscore', 'minmax', 'max', 'mean'
    RandomFlip,              # axes=[0,1,2], prob=0.5
    RandomRotation90,        # axes=(0,1), prob=0.5
    RandomNoise,             # std=0.01, prob=0.5
    ToTensor                 # Convert to PyTorch tensor
)

transforms = Compose([
    IntensityNormalization(method='zscore'),
    RandomFlip(axes=[1, 2], prob=0.5),
    RandomRotation90(axes=(1, 2), prob=0.3),
    RandomNoise(std=0.01, prob=0.2)
])
```

## CSV Format

### Simple Format

```csv
id,image,roi,label
sub001,/data/sub001/T1.nii.gz,/data/sub001/roi.nii.gz,1.0
sub002,/data/sub002/T1.nii.gz,/data/sub002/roi.nii.gz,0.0
```

### Multi-Image Format

```csv
id,T1,T2,FLAIR,segmentation,label
sub001,/data/sub001/T1.nii.gz,/data/sub001/T2.nii.gz,/data/sub001/FLAIR.nii.gz,/data/sub001/seg.nii.gz,1.0
```

### Legacy Format (still supported)

```csv
label,image1,image2
1.0,/data/sub001/T1.nii.gz,/data/sub001/T2.nii.gz
0.0,/data/sub002/T1.nii.gz,/data/sub002/T2.nii.gz
```

## Batch Output Format

```python
batch = dataset[idx]

# Dict with keys:
{
    'images': torch.Tensor,     # [C, D, H, W] or [C, H, W]
    'rois': torch.Tensor,       # [1, D, H, W] or None
    'labelmaps': torch.Tensor,  # [1, D, H, W] or None
    'label': torch.Tensor,      # Scalar classification label
    'id': str,                  # Subject identifier
    'meta': dict               # If return_meta=True
}
```

## DataLoader Configuration

### For Training

```python
from torch.utils.data import DataLoader

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    num_workers=4,           # Parallel loading
    pin_memory=True,         # Faster GPU transfer
    persistent_workers=True  # Keep workers alive
)
```

### For Validation/Testing

```python
loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=False,
    num_workers=4
)
```

## Performance Tips

1. **Enable caching** - 50x speedup:
   ```python
   dataset = TrenoDataset(..., cache_dir='./cache')
   ```

2. **Use multiple workers**:
   ```python
   loader = DataLoader(dataset, num_workers=4)
   ```

3. **Pin memory for GPU**:
   ```python
   loader = DataLoader(dataset, pin_memory=True)
   ```

4. **Adjust target_size** if memory issues:
   ```python
   dataset = TrenoDataset(..., target_size=[32, 32, 32])
   ```

## Legacy API (Backward Compatible)

```python
from treno.loaders import ImageLabelmapDataset

# Old code still works!
all_transforms = {'resize': [64, 64, 64], 'normalizex': 'max'}

dataset = ImageLabelmapDataset(
    'train.csv',
    transform=all_transforms,
    index=[0, 1, 2]
)

loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=2,
    shuffle=True
)

for x, y in loader:
    # x: [B, 1, D, H, W]
    # y: [B, C, D, H, W] where C = number of classes
    pass
```

## Documentation

- **Quick Start**: This file
- **Migration Guide**: `MIGRATION_GUIDE.md`
- **Full Documentation**: `README.md`
- **Examples**: `examples/example_new_loader.py`
- **pyable-dataloader docs**: `/home/erosm/packages/pyable-dataloader/README.md`

## Common Issues

### Import Error: "pyable-dataloader not available"

```bash
cd /home/erosm/packages/pyable-dataloader
pip install -e .
```

### Slow Loading

Enable caching:
```python
dataset = TrenoDataset(..., cache_dir='./cache')
```

### Out of Memory

Reduce batch size or target size:
```python
dataset = TrenoDataset(..., target_size=[32, 32, 32])
loader = DataLoader(dataset, batch_size=2)
```

### Labels Interpolated (Non-Integer Values)

This should NOT happen with TrenoDataset. If it does, please report it.
The new dataset automatically uses nearest-neighbor for labelmaps.

## Getting Help

1. Check `MIGRATION_GUIDE.md` for detailed instructions
2. Run `python examples/example_new_loader.py` to see working examples
3. Read pyable-dataloader documentation for advanced features

## Summary

**For new projects**: Use modern API with `TrenoDataset`  
**For existing projects**: Legacy API still works, migrate when ready  
**Performance**: Enable caching for 50x speedup  
**Augmentation**: Use `Compose` with transform classes  
**Best practice**: Use `cache_dir` and `num_workers` for speed
