# Migration Guide: Old Loaders → New TrenoDataset

This guide helps you migrate from the legacy `ImageImageDataset` and `ImageLabelmapDataset` classes to the new `TrenoDataset` based on `pyable-dataloader`.

## Why Migrate?

✅ **Proper pyable v3 integration** - Correct ZYX array conventions  
✅ **Automatic label preservation** - No interpolated label values  
✅ **Better performance** - Smart caching system  
✅ **More flexible** - Supports JSON, CSV, multi-CSV formats  
✅ **Modular transforms** - Composable augmentation pipeline  
✅ **Better documentation** - Clear API and examples  

## Quick Comparison

### Old Way (Legacy)

```python
from treno.loaders import ImageLabelmapDataset

# Legacy class with custom transforms dict
all_transforms = {'resize': [320, 320, 120], 'normalizex': 'max'}

dataset = ImageLabelmapDataset(
    annotations_file='train.csv',
    transform=all_transforms,
    index=[0, 1, 2]
)

loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=2,
    shuffle=True
)
```

### New Way (Modern)

```python
from treno.loaders import create_treno_dataset_from_csv
from pyable_dataloader import Compose, IntensityNormalization, RandomFlip
from torch.utils.data import DataLoader

# Create dataset with modern transforms
dataset = create_treno_dataset_from_csv(
    csv_file='train.csv',
    target_size=[64, 64, 64],
    target_spacing=2.0,
    augmentation=True,
    cache_dir='./cache'
)

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)
```

## Step-by-Step Migration

### Step 1: Install pyable-dataloader

```bash
cd /home/erosm/packages/pyable-dataloader
pip install -e .
```

### Step 2: Convert Your CSV Files

Your old CSV format:
```csv
label,image1,image2
1.0,/data/sub001/T1.nii.gz,/data/sub001/T2.nii.gz
0.0,/data/sub002/T1.nii.gz,/data/sub002/T2.nii.gz
```

Convert to manifest:

```python
from treno.loaders import create_manifest_from_csv, save_manifest

# Convert CSV to manifest dict
manifest = create_manifest_from_csv(
    csv_file='train.csv',
    image_columns=['image1', 'image2'],  # Specify image columns
    label_column='label'
)

# Save to JSON (optional but recommended)
save_manifest(manifest, 'train_manifest.json')
```

Or use the convenience function:

```python
from treno.loaders import create_treno_dataset_from_csv

# One-liner: creates dataset directly from CSV
dataset = create_treno_dataset_from_csv(
    csv_file='train.csv',
    target_size=[64, 64, 64],
    target_spacing=2.0,
    augmentation=True,
    image_columns=['image1', 'image2'],
    label_column='label'
)
```

### Step 3: Update Your Training Code

**Old code:**

```python
from treno.loaders import ImageLabelmapDataset

all_transforms = {'resize': [320, 320, 120], 'normalizex': 'max'}

train_dataset = ImageLabelmapDataset(
    'train.csv',
    transform=all_transforms,
    index=[0, 1, 2]
)

test_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=2,
    shuffle=False
)

for x, y in test_loader:
    # x: [B, 1, D, H, W]
    # y: [B, C, D, H, W] where C = number of classes
    outputs = model(x)
    loss = criterion(outputs, y)
```

**New code:**

```python
from treno.loaders import TrenoDataset, create_manifest_from_csv
from pyable_dataloader import Compose, IntensityNormalization, RandomFlip

# Create manifest
manifest = create_manifest_from_csv(
    'train.csv',
    labelmap_column='labelmap',  # or roi_column if using ROIs
    label_column='label'
)

# Define transforms
transforms = Compose([
    IntensityNormalization(method='max'),  # Replaces 'normalizex': 'max'
])

# Create dataset
train_dataset = TrenoDataset(
    manifest=manifest,
    target_size=[320, 320, 120],  # Replaces 'resize'
    target_spacing=2.0,
    transforms=transforms,
    cache_dir='./cache'
)

# Create DataLoader
test_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=2,
    shuffle=False,
    num_workers=4  # Speed up loading!
)

# Training loop
for batch in test_loader:
    images = batch['images']  # [B, C, D, H, W]
    labelmaps = batch['labelmaps']  # [B, 1, D, H, W]
    labels = batch['label']  # [B]
    
    outputs = model(images)
    
    # For segmentation
    if labelmaps is not None:
        loss = criterion(outputs, labelmaps)
    # For classification
    else:
        loss = criterion(outputs, labels)
```

### Step 4: Update Transform Configurations

**Old transform dict:**

```python
all_transforms = {
    'resize': [320, 320, 120],
    'normalizex': 'max',
    'normalizey': 'z'
}
```

**New Compose transforms:**

```python
from pyable_dataloader import (
    Compose,
    IntensityNormalization,
    RandomFlip,
    RandomRotation90,
    RandomNoise
)

transforms = Compose([
    IntensityNormalization(method='max'),      # Replaces 'normalizex': 'max'
    RandomFlip(axes=[1, 2], prob=0.5),        # Add augmentation
    RandomRotation90(axes=(1, 2), prob=0.3),  # Add augmentation
    RandomNoise(std=0.01, prob=0.2)           # Add augmentation
])

dataset = TrenoDataset(
    manifest=manifest,
    target_size=[320, 320, 120],  # Replaces 'resize'
    transforms=transforms
)
```

### Step 5: Handle Multi-Channel Label Maps

**Old way:**

```python
from treno.loaders import labelMapToChannel

dataset = ImageLabelmapDataset(
    'train.csv',
    transform=transforms,
    index=[0, 1, 2]  # Convert labels 0,1,2 to separate channels
)
```

**New way:**

```python
# Option 1: Keep as single-channel labelmap (recommended)
dataset = TrenoDataset(
    manifest=manifest,
    target_size=[64, 64, 64]
)

# In your model, handle multi-class labels:
for batch in loader:
    labelmaps = batch['labelmaps']  # [B, 1, D, H, W] with values [0, 1, 2]
    
    # Use CrossEntropyLoss or similar
    loss = nn.CrossEntropyLoss()(outputs, labelmaps.squeeze(1).long())

# Option 2: Convert to one-hot in training loop
from treno.loaders import labelMapToChannel

for batch in loader:
    labelmaps = batch['labelmaps'].numpy()  # [B, 1, D, H, W]
    
    # Convert each sample to one-hot
    labelmaps_onehot = np.stack([
        labelMapToChannel(lm[0], include=[0, 1, 2])
        for lm in labelmaps
    ])  # [B, 3, D, H, W]
    
    labelmaps_onehot = torch.from_numpy(labelmaps_onehot)
    loss = criterion(outputs, labelmaps_onehot)
```

## Common Migration Patterns

### Pattern 1: Simple Image-to-Image Training

```python
# OLD
from treno.loaders import ImageImageDataset

dataset = ImageImageDataset('train.csv', transform={'normalizex': 'max'})

# NEW
from treno.loaders import create_treno_dataset_from_csv

dataset = create_treno_dataset_from_csv(
    'train.csv',
    target_spacing=2.0,
    augmentation=False
)
```

### Pattern 2: Image-to-Labelmap Segmentation

```python
# OLD
from treno.loaders import ImageLabelmapDataset

dataset = ImageLabelmapDataset(
    'train.csv',
    transform={'resize': [64, 64, 64], 'normalizex': 'z'},
    index=[0, 1, 2, 3]
)

# NEW
from treno.loaders import TrenoDataset, create_manifest_from_csv
from pyable_dataloader import Compose, IntensityNormalization

manifest = create_manifest_from_csv(
    'train.csv',
    labelmap_column='labelmap',
    label_column='label'
)

transforms = Compose([
    IntensityNormalization(method='zscore')
])

dataset = TrenoDataset(
    manifest=manifest,
    target_size=[64, 64, 64],
    transforms=transforms,
    cache_dir='./cache'
)
```

### Pattern 3: Multi-Modal with Augmentation

```python
# OLD
from treno.loaders import ImageLabelmapDataset

all_transforms = {
    'resize': [128, 128, 64],
    'normalizex': 'max'
}

dataset = ImageLabelmapDataset('train.csv', transform=all_transforms)

# NEW
from treno.loaders import TrenoDataset, create_manifest_from_csv
from pyable_dataloader import Compose, IntensityNormalization, RandomFlip

manifest = create_manifest_from_csv(
    'train.csv',
    image_columns=['T1', 'T2', 'FLAIR'],
    labelmap_column='segmentation',
    label_column='label'
)

transforms = Compose([
    IntensityNormalization(method='max'),
    RandomFlip(axes=[1, 2], prob=0.5)
])

dataset = TrenoDataset(
    manifest=manifest,
    target_size=[128, 128, 64],
    target_spacing=1.5,
    transforms=transforms,
    stack_channels=True,  # Stack T1, T2, FLAIR as channels
    cache_dir='./cache'
)
```

### Pattern 4: ROI-Based Training

```python
# NEW - ROI masking and centering
from treno.loaders import TrenoDataset, create_manifest_from_csv

manifest = create_manifest_from_csv(
    'train.csv',
    image_columns=['image'],
    roi_column='roi',
    label_column='label'
)

dataset = TrenoDataset(
    manifest=manifest,
    target_size=[64, 64, 64],
    roi_mask=True,  # Multiply image by ROI
    roi_center=True,  # Center volume around ROI
    roi_dilation=5.0,  # Dilate ROI by 5mm
    cache_dir='./cache'
)
```

## Backward Compatibility

Don't want to migrate yet? The legacy classes are still available:

```python
from treno.loaders import (
    ImageImageDataset,
    ImageLabelmapDataset,
    ImaImaDataset,
    ImaRoiDataset
)

# Old code still works!
dataset = ImageLabelmapDataset('train.csv', transform={'normalizex': 'max'})
```

But we recommend migrating when possible for better performance and features.

## Performance Tips

1. **Enable caching** for 50x faster loading after first epoch:
   ```python
   dataset = TrenoDataset(..., cache_dir='./cache')
   ```

2. **Use multiple workers**:
   ```python
   loader = DataLoader(dataset, num_workers=4, persistent_workers=True)
   ```

3. **Pin memory for GPU training**:
   ```python
   loader = DataLoader(dataset, pin_memory=True)
   ```

4. **Reduce target_size** if running out of memory:
   ```python
   dataset = TrenoDataset(..., target_size=[32, 32, 32])
   ```

## Troubleshooting

### "pyable-dataloader not available"

Install the package:
```bash
cd /home/erosm/packages/pyable-dataloader
pip install -e .
```

### "Labels are interpolated"

This should NOT happen with the new dataset! The new implementation automatically uses nearest-neighbor interpolation for labelmaps. If you see this, please report it.

### "Slow loading on first epoch"

This is normal without caching. Enable caching:
```python
dataset = TrenoDataset(..., cache_dir='./cache')
```

Second epoch will be ~50x faster!

### "Different tensor shapes than before"

Old format: `x, y = dataset[idx]`  
New format: `batch = dataset[idx]` where batch is a dict with keys `'images'`, `'labelmaps'`, `'label'`, etc.

Update your code:
```python
# OLD
x, y = dataset[idx]

# NEW
batch = dataset[idx]
x = batch['images']
y = batch['labelmaps']  # or batch['label'] for classification
```

## Need Help?

- See `/home/erosm/packages/pyable-dataloader/README.md` for full documentation
- Check examples in `/home/erosm/packages/pyable-dataloader/examples/`
- Read the pyable-dataloader QUICKSTART.md for quick recipes

## Summary

**Key Changes:**
1. ✅ Use `TrenoDataset` instead of `ImageImageDataset`/`ImageLabelmapDataset`
2. ✅ Convert CSV to manifest with `create_manifest_from_csv()`
3. ✅ Use `Compose` transforms instead of dict-based transforms
4. ✅ Access data via dict keys: `batch['images']`, `batch['labelmaps']`, `batch['label']`
5. ✅ Enable caching with `cache_dir='./cache'`
6. ✅ Use `num_workers` in DataLoader for speed

**Benefits:**
- 🚀 50x faster with caching
- ✅ Automatic label preservation
- 🎯 Correct pyable v3 conventions
- 🔧 More flexible and maintainable
- 📚 Better documented
