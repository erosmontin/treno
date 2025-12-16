# Treno: LLM Agent Pipeline Builder Guide

**Comprehensive reference for LLM agents building deep learning pipelines with Treno v3.5.0**

> **Purpose**: This guide provides all essential information for LLM agents to autonomously build, configure, and debug medical imaging deep learning pipelines using the Treno package.

---

## Table of Contents

1. [Package Overview](#package-overview)
2. [Task Selection Matrix](#task-selection-matrix)
3. [Model Architecture Guide](#model-architecture-guide)
4. [Data Loading Pipeline](#data-loading-pipeline)
5. [Training Configuration](#training-configuration)
6. [Complete Pipeline Examples](#complete-pipeline-examples)
7. [Loss Functions Reference](#loss-functions-reference)
8. [Common Patterns & Best Practices](#common-patterns--best-practices)
9. [Debugging Guide](#debugging-guide)
10. [API Reference](#api-reference)

---

## Package Overview

### What is Treno?

Treno is a PyTorch-based framework for medical imaging deep learning that provides:
- **ND Support**: 1D (signals), 2D (images), 3D (volumes)
- **4 Task Types**: Classification, Regression, Segmentation, Map-to-Map
- **Multiple Architectures**: U-Net, U-Net++, ResNet, LeNet variants
- **Built-in Features**: Radiomics, attention mechanisms, metadata injection
- **Modern Data Loading**: PyTorch DataLoader with smart caching

### Key Imports

```python
# Models
from treno import (
    EMUNet,              # Standard U-Net (1D/2D/3D)
    EMUNetPP,            # U-Net++ with dense skip connections
    EMUNet1D,            # 1D U-Net with radiomics
    UNet1DOptimized,     # Dilated 1D U-Net
    EMUNetMapToMap,      # U-Net for image translation
    EMUNetPPMapToMap,    # U-Net++ for image translation
    EMResNet,            # ResNet with radiomics
    EMLeNet,             # LeNet-style CNN
)

# Data Loading
from treno import (
    create_treno_dataset_from_csv,
    TrenoDataset,
    create_manifest_from_csv,
)

# Utilities
from treno.utils import (
    # Explainability
    GradCAM,
    compute_saliency_map,
    postprocess_cam,
    # Metrics
    compute_metrics,
    compute_binary_metrics,
    compute_multilabel_sensitivity_specificity,
    # Data splitting (medical imaging aware)
    stratified_group_split,
    extract_patient_groups,
    # Feature selection
    feature_selection,
    filterFeaturesByCorrelation,
    # Visualization
    visualize_embeddings,
    write_confusion_matrix_to_tensorboard,
)

# Custom Loss Functions (Segmentation)
from treno.losses import (
    EMLabelMapLoss,       # Metric-based segmentation loss
    dice_loss3D,          # Dice loss for 3D
    jacard_loss3D,        # Jaccard/IoU loss for 3D
    EMMulticlassLoss,     # Combined CE + Dice + Jaccard
)

# Building blocks (advanced)
from treno import (
    MapToMapHead,
    NetworkHead,
    CBAM,
    SkipConnectionAligner,
)
```

---

## Task Selection Matrix

### When to Use Each Task Type

| Task | Use When | Input/Output | Loss Function |
|------|----------|--------------|---------------|
| **Classification** | Predicting discrete categories | Image → Class probabilities | `BCEWithLogitsLoss`, `CrossEntropyLoss` |
| **Regression** | Predicting continuous values | Image → Scalar values | `MSELoss`, `L1Loss`, `HuberLoss` |
| **Segmentation** | Pixel-wise labeling | Image → Labeled mask | `CrossEntropyLoss`, `DiceLoss` |
| **Map-to-Map** | Image-to-image translation | Image → Reconstructed image | `L1Loss`, `MSELoss`, `PerceptualLoss` |

### Dimension Selection

```python
# 1D: Time-series, signals, sequences
dimension = 1  # Input shape: [B, C, L]

# 2D: Images, slices, radiographs
dimension = 2  # Input shape: [B, C, H, W]

# 3D: Volumes, CT, MRI
dimension = 3  # Input shape: [B, C, D, H, W]
```

### Task Compatibility

| Model | Classification | Regression | Segmentation | Map-to-Map |
|-------|---------------|------------|--------------|------------|
| `EMUNet` | ❌ | ❌ | ✅ 1D/2D/3D | ❌ |
| `EMUNetPP` | ❌ | ❌ | ✅ 1D/2D/3D | ❌ |
| `EMUNet1D` | ✅ 1D only | ✅ 1D only | ✅ 1D only | ❌ |
| `UNet1DOptimized` | ✅ 1D only | ✅ 1D only | ❌ | ✅ 1D only |
| `EMUNetMapToMap` | ❌ | ❌ | ❌ | ✅ 1D/2D/3D |
| `EMUNetPPMapToMap` | ❌ | ❌ | ❌ | ✅ 1D/2D/3D |
| `EMResNet` | ✅ 1D/2D/3D | ✅ 1D/2D/3D | ❌ | ❌ |
| `EMLeNet` | ✅ 1D/2D/3D | ✅ 1D/2D/3D | ❌ | ❌ |

---

## Deep Radiomics (Feature Extraction)

After training any model, extract pooled deep features and optionally concatenate engineered radiomics via a single helper:

```python
from treno import get_deep_radiomics_features, EMUNet
import torch

model = EMUNet(1, 4, 3, use_radiomics=True).eval()
x = torch.randn(2, 1, 64, 64, 64)

# Returns [B, F] matrix: deep pooled features (+ engineered radiomics if enabled)
features = get_deep_radiomics_features(model, x, pool='avg', include_engineered=True)
```

Notes:
- Works across all models (U-Net/UNet++, MapToMap, LeNet, ResNet) and in 1D/2D/3D.
- `pool='avg'|'max'` controls spatial pooling of deep features.
- Set `include_engineered=False` to return only deep features.

---

## Model Architecture Guide

### Standard U-Net: `EMUNet`

**Best for**: Semantic segmentation (pixel/voxel-level labeling)

> **Note**: EMUNet is segmentation-only. For classification/regression, use `EMLeNet` or `EMResNet`.

```python
from treno import EMUNet

model = EMUNet(
    in_channels=1,              # Input channels (1 for grayscale, 3 for RGB)
    out_channels=4,             # Number of segmentation classes
    dimension=3,                # 1, 2, or 3
    
    # Optional features
    use_attention=True,         # Add CBAM attention blocks
    use_radiomics=False,        # Extract handcrafted features
    extra_params_dim=0,         # Extra metadata dimension (age, sex, etc.)
    
    # Architecture config
    num_filters=[64, 128, 256],  # Filter counts per level
    dropout_rate=0.0,           # Dropout rate
    use_skip_attention=False,   # Attention gates on skip connections
    
    # Radiomics config (if use_radiomics=True)
    radii=[1],                  # GLCM radii (multi-scale)
    num_bins=256,               # Histogram bins for texture
)
```

**Output shape**: `[B, out_channels, D, H, W]` - raw logits (use with `CrossEntropyLoss`)

### U-Net++: `EMUNetPP`

**Best for**: High-quality segmentation with dense skip connections

> **Note**: EMUNetPP is segmentation-only. For classification/regression, use `EMLeNet` or `EMResNet`.

```python
from treno import EMUNetPP

model = EMUNetPP(
    in_channels=1,
    out_channels=5,             # Number of segmentation classes
    dimension=3,
    use_attention=True,
    use_radiomics=True,
    extra_params_dim=3,
)
```

**Advantages**: Dense skip connections improve gradient flow and feature reuse.

### Map-to-Map U-Net: `EMUNetMapToMap`

**Best for**: Image denoising, synthesis, style transfer, reconstruction

```python
from treno import EMUNetMapToMap

model = EMUNetMapToMap(
    in_channels=1,              # Input image channels
    out_channels=1,             # Output image channels
    dimension=2,                # 1, 2, or 3
    
    # Optional
    use_attention=True,
    num_filters=[64, 128, 256, 512],
    depth=4,
    dropout=0.0,
    
    # Map-to-map specific
    output_activation='tanh',   # 'sigmoid', 'tanh', or None
)
```

**Output**: `[B, out_channels, D, H, W]` - reconstructed image

**Use with**: `L1Loss`, `MSELoss`, or perceptual losses

### Map-to-Map U-Net++: `EMUNetPPMapToMap`

```python
from treno import EMUNetPPMapToMap

model = EMUNetPPMapToMap(
    # Same as EMUNetMapToMap
    in_channels=1,
    out_channels=1,
    dimension=3,
    output_activation='sigmoid',  # For normalized outputs [0, 1]
)
```

**Advantages**: Superior reconstruction quality due to dense skip connections.

### 1D Time-Series: `EMUNet1D`

**Best for**: Time-series classification with radiomics

```python
from treno import EMUNet1D

model = EMUNet1D(
    in_channels=1,              # Univariate or multivariate
    out_channels=5,             # Classes or regression outputs
    task='classification',      # 'classification' or 'regression'
    
    use_radiomics=True,         # Extract temporal features
    extra_params_dim=3,         # Clinical metadata
    
    num_filters=[64, 128, 256],
    depth=3,
    dropout=0.1,
)
```

**Input**: `[B, C, L]` where `L` is sequence length  
**Output**: `[B, out_channels]`

### 1D Optimized: `UNet1DOptimized`

**Best for**: Variable-length sequences, map-to-map tasks

```python
from treno import UNet1DOptimized

model = UNet1DOptimized(
    in_channels=1,
    out_channels=1,
    task='maptomap',            # 'classification', 'regression', 'maptomap'
    
    # 1D-specific
    use_dilated=True,           # Dilated convolutions
    dilation_schedule=[1, 2, 4, 8],  # Dilation rates
    pooling_type='max',         # 'max', 'avg', 'adaptive'
    
    num_filters=[64, 128, 256],
    depth=3,
)
```

**Advantages**: Large receptive field without excessive downsampling.

### ResNet: `EMResNet`

**Best for**: Deep feature learning, classification/regression

```python
from treno import EMResNet

model = EMResNet(
    in_channels=3,
    out_channels=1000,
    dimension=2,
    task='classification',
    
    use_radiomics=True,
    extra_params_dim=5,
    
    num_blocks=[3, 4, 6, 3],    # ResNet-50 style
    num_filters=[64, 128, 256, 512],
)
```

### LeNet: `EMLeNet`

**Best for**: Small images, lightweight models

```python
from treno import EMLeNet

model = EMLeNet(
    in_channels=1,
    out_channels=10,
    dimension=2,
    task='classification',
)
```

---

## Data Loading Pipeline

### Quick Setup from CSV

**CSV Format**:
```csv
image_path,label
/path/to/scan1.nii.gz,0
/path/to/scan2.nii.gz,1
```

**Code**:
```python
from treno import create_treno_dataset_from_csv
from torch.utils.data import DataLoader

# One-line dataset creation
dataset = create_treno_dataset_from_csv(
    csv_file='train.csv',
    target_size=[128, 128, 128],    # Resample to this size
    target_spacing=2.0,              # Resample to this spacing (mm)
    augmentation=True,               # Enable random flips
    cache_dir='./cache',             # Cache processed images
)

# PyTorch DataLoader
loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)

# Iterate
for batch in loader:
    images = batch['images']    # [B, C, D, H, W]
    labels = batch['label']     # [B]
```

### Multi-Channel Input (Multi-Modal)

**CSV Format**:
```csv
T1,T2,FLAIR,label
/path/t1.nii.gz,/path/t2.nii.gz,/path/flair.nii.gz,1
```

**Code**:
```python
dataset = create_treno_dataset_from_csv(
    csv_file='train.csv',
    target_size=[128, 128, 128],
    target_spacing=1.5,
    augmentation=True,
    image_columns=['T1', 'T2', 'FLAIR'],  # Specify column names
    label_column='label',
)

# Output: batch['images'] shape = [B, 3, 128, 128, 128]
```

### Segmentation Data

**CSV Format**:
```csv
image,segmentation
/path/image.nii.gz,/path/mask.nii.gz
```

**Code**:
```python
dataset = create_treno_dataset_from_csv(
    csv_file='train.csv',
    target_size=[128, 128, 128],
    target_spacing=2.0,
    augmentation=True,
    labelmap_column='segmentation',  # For segmentation masks
)

# Output: batch['labelmap'] shape = [B, 1, 128, 128, 128]
```

### Custom Transforms

```python
from pyable_dataloader import Compose, IntensityNormalization, RandomFlip, RandomRotation90

transforms = Compose([
    IntensityNormalization(mode='z-score'),
    RandomFlip(axis=0, prob=0.5),
    RandomFlip(axis=1, prob=0.5),
    RandomRotation90(axis=(0, 1), k=None),  # Random 90° rotations
])

dataset = create_treno_dataset_from_csv(
    csv_file='train.csv',
    target_size=[128, 128, 128],
    target_spacing=2.0,
    custom_transforms=transforms,
)
```

### Advanced: Manifest-Based Loading

```python
from treno import create_manifest_from_csv, TrenoDataset, save_manifest

# Step 1: Create manifest
manifest = create_manifest_from_csv(
    csv_file='train.csv',
    image_columns=['T1', 'T2'],
    label_column='diagnosis',
    labelmap_column='mask',
)

# Step 2: Save manifest (optional, for reuse)
save_manifest(manifest, 'train_manifest.json')

# Step 3: Create dataset
dataset = TrenoDataset(
    manifest=manifest,
    target_size=[64, 64, 64],
    target_spacing=2.0,
    augmentation=True,
)
```

---

## Training Configuration

### Complete Classification Pipeline

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from treno import EMLeNet, create_treno_dataset_from_csv

# 1. Data
train_dataset = create_treno_dataset_from_csv(
    'train.csv',
    target_size=[128, 128, 128],
    target_spacing=2.0,
    augmentation=True,
)
val_dataset = create_treno_dataset_from_csv(
    'val.csv',
    target_size=[128, 128, 128],
    target_spacing=2.0,
    augmentation=False,
)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)

# 2. Model (use EMLeNet for classification, NOT EMUNet)
model = EMLeNet(
    in_channels=1,
    out_channels=5,  # 5 classes
    dimension=3,
    task='classification',
    use_attention=True,
    use_radiomics=True,
).cuda()

# 3. Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

# 4. Training Loop
for epoch in range(100):
    model.train()
    for batch in train_loader:
        images = batch['images'].cuda()
        labels = batch['label'].cuda()
        
        # Forward
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    scheduler.step()
    
    # Validation
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            images = batch['images'].cuda()
            labels = batch['label'].cuda()
            outputs = model(images)
            val_loss = criterion(outputs, labels)
```

### Segmentation Pipeline

```python
from treno import EMUNetPP

# Model
model = EMUNetPP(
    in_channels=1,
    out_channels=4,  # 4 segmentation classes (including background)
    dimension=3,
    task='segmentation',
    use_attention=True,
).cuda()

# Loss (CrossEntropyLoss expects class indices, not one-hot)
criterion = nn.CrossEntropyLoss()

# Training
for batch in train_loader:
    images = batch['images'].cuda()      # [B, 1, D, H, W]
    masks = batch['labelmap'].cuda()     # [B, 1, D, H, W] - integer labels
    
    outputs = model(images)               # [B, 4, D, H, W] - logits
    
    # CrossEntropyLoss expects [B, C, D, H, W] and [B, D, H, W]
    loss = criterion(outputs, masks.squeeze(1).long())
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### Map-to-Map Pipeline (Image Denoising)

```python
from treno import EMUNetMapToMap

# Model
model = EMUNetMapToMap(
    in_channels=1,
    out_channels=1,
    dimension=2,
    output_activation='sigmoid',  # Output in [0, 1]
).cuda()

# Loss
criterion = nn.L1Loss()  # MAE

# Training
for batch in train_loader:
    noisy = batch['noisy_images'].cuda()
    clean = batch['clean_images'].cuda()
    
    reconstructed = model(noisy)
    loss = criterion(reconstructed, clean)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### With Extra Parameters (Metadata)

```python
# Model
model = EMUNet(
    in_channels=1,
    out_channels=1,
    dimension=3,
    task='regression',
    extra_params_dim=5,  # age, sex, BMI, TR, TE
).cuda()

# Training
for batch in train_loader:
    images = batch['images'].cuda()
    metadata = batch['metadata'].cuda()  # [B, 5]
    targets = batch['target'].cuda()
    
    outputs = model(images, extra_params=metadata)
    loss = criterion(outputs, targets)
```

---

## Loss Functions Reference

### Classification

**Binary Classification (2 classes)**:
```python
criterion = nn.BCEWithLogitsLoss()  # Model outputs raw logits
# OR
criterion = nn.BCELoss()  # Model outputs sigmoid probabilities
```

**Multi-Class Classification**:
```python
criterion = nn.CrossEntropyLoss()  # Model outputs logits [B, C]
```

**With Class Weights** (for imbalanced data):
```python
class_weights = torch.tensor([1.0, 3.0, 5.0])  # Higher weight for rare classes
criterion = nn.CrossEntropyLoss(weight=class_weights)
```

### Regression

```python
# L2 Loss (MSE)
criterion = nn.MSELoss()

# L1 Loss (MAE) - more robust to outliers
criterion = nn.L1Loss()

# Huber Loss - combination of L1 and L2
criterion = nn.HuberLoss(delta=1.0)
```

### Segmentation

```python
# Standard
criterion = nn.CrossEntropyLoss()

# With class weights
weights = torch.tensor([0.1, 1.0, 2.0, 3.0])  # Lower weight for background
criterion = nn.CrossEntropyLoss(weight=weights)

# Dice Loss (requires custom implementation or library)
# from monai.losses import DiceLoss
# criterion = DiceLoss(softmax=True)
```

### Map-to-Map

```python
# L1 (MAE) - sharper results
criterion = nn.L1Loss()

# L2 (MSE) - smoother results
criterion = nn.MSELoss()

# Perceptual Loss (requires pretrained VGG)
# from torchvision.models import vgg16
# Use VGG features for comparison
```

---

## Complete Pipeline Examples

### Example 1: 3D Brain MRI Classification

**Goal**: Classify brain MRI scans into 5 disease categories

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from treno import EMLeNet, create_treno_dataset_from_csv

# Data
train_dataset = create_treno_dataset_from_csv(
    'brain_train.csv',
    target_size=[128, 128, 128],
    target_spacing=1.5,
    augmentation=True,
)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4)

# Model (use EMLeNet for classification)
model = EMLeNet(
    in_channels=1,
    out_channels=5,
    dimension=3,
    task='classification',
    use_attention=True,
    use_radiomics=True,
    num_filters=[32, 64, 128, 256],
).cuda()

# Training
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

for epoch in range(50):
    for batch in train_loader:
        images = batch['images'].cuda()
        labels = batch['label'].cuda()
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### Example 2: 2D Chest X-Ray Multi-Label Classification

**Goal**: Predict multiple pathologies from chest X-rays

```python
from treno import EMLeNet

# Model for multi-label (each class independent, use EMLeNet for classification)
model = EMLeNet(
    in_channels=1,
    out_channels=14,  # 14 pathologies
    dimension=2,
    task='classification',
    use_attention=True,
).cuda()

# Binary cross-entropy for each label
criterion = nn.BCEWithLogitsLoss()

for batch in train_loader:
    images = batch['images'].cuda()    # [B, 1, 512, 512]
    labels = batch['labels'].cuda()    # [B, 14] - multi-hot encoding
    
    outputs = model(images)             # [B, 14] - logits
    loss = criterion(outputs, labels.float())
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### Example 3: 3D Tumor Segmentation

**Goal**: Segment brain tumors into 4 classes (background, edema, core, enhancing)

```python
from treno import EMUNetPP

model = EMUNetPP(
    in_channels=4,  # T1, T1ce, T2, FLAIR
    out_channels=4,  # 4 segmentation classes
    dimension=3,
    use_attention=True,  # EMUNetPP is segmentation-only, no task param needed
).cuda()

criterion = nn.CrossEntropyLoss()

for batch in train_loader:
    images = batch['images'].cuda()      # [B, 4, 128, 128, 128]
    masks = batch['labelmap'].cuda()     # [B, 1, 128, 128, 128]
    
    outputs = model(images)               # [B, 4, 128, 128, 128]
    loss = criterion(outputs, masks.squeeze(1).long())
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### Example 4: 2D Image Denoising

**Goal**: Remove noise from 2D medical images

```python
from treno import EMUNetMapToMap

model = EMUNetMapToMap(
    in_channels=1,
    out_channels=1,
    dimension=2,
    output_activation='sigmoid',
).cuda()

criterion = nn.L1Loss()

for batch in train_loader:
    noisy = batch['noisy'].cuda()
    clean = batch['clean'].cuda()
    
    reconstructed = model(noisy)
    loss = criterion(reconstructed, clean)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### Example 5: 1D ECG Classification

**Goal**: Classify ECG signals into arrhythmia categories

```python
from treno import EMUNet1D

model = EMUNet1D(
    in_channels=12,  # 12-lead ECG
    out_channels=5,   # 5 arrhythmia types
    task='classification',
    use_radiomics=True,
).cuda()

criterion = nn.CrossEntropyLoss()

for batch in train_loader:
    ecg = batch['signal'].cuda()     # [B, 12, 5000]
    labels = batch['label'].cuda()
    
    outputs = model(ecg)
    loss = criterion(outputs, labels)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

---

## Common Patterns & Best Practices

### Pattern 1: Learning Rate Scheduling

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Cosine annealing
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

# Reduce on plateau
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=10
)

# Step decay
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# Use in training loop
for epoch in range(num_epochs):
    train(...)
    val_loss = validate(...)
    scheduler.step()  # For CosineAnnealing/StepLR
    # scheduler.step(val_loss)  # For ReduceLROnPlateau
```

### Pattern 2: Early Stopping

```python
best_val_loss = float('inf')
patience = 10
counter = 0

for epoch in range(num_epochs):
    train_loss = train(...)
    val_loss = validate(...)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pth')
        counter = 0
    else:
        counter += 1
    
    if counter >= patience:
        print("Early stopping!")
        break
```

### Pattern 3: Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in train_loader:
    images = batch['images'].cuda()
    labels = batch['label'].cuda()
    
    optimizer.zero_grad()
    
    # Forward with autocast
    with autocast():
        outputs = model(images)
        loss = criterion(outputs, labels)
    
    # Backward with scaler
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### Pattern 4: Gradient Clipping

```python
for batch in train_loader:
    optimizer.zero_grad()
    loss = ...
    loss.backward()
    
    # Clip gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
```

### Pattern 5: Model Checkpointing

```python
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'loss': loss,
}
torch.save(checkpoint, f'checkpoint_epoch_{epoch}.pth')

# Load
checkpoint = torch.load('checkpoint_epoch_50.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
```

---

## Debugging Guide

### Issue 1: Out of Memory (OOM)

**Symptoms**: CUDA out of memory error

**Solutions**:
```python
# 1. Reduce batch size
batch_size = 1  # Start small

# 2. Reduce model size
num_filters = [32, 64, 128]  # Instead of [64, 128, 256, 512]
depth = 3  # Instead of 4

# 3. Reduce input size
target_size = [64, 64, 64]  # Instead of [128, 128, 128]

# 4. Disable radiomics
use_radiomics = False

# 5. Use gradient checkpointing (advanced)
# torch.utils.checkpoint.checkpoint(module, inputs)
```

### Issue 2: Slow Training

**Solutions**:
```python
# 1. Increase num_workers
loader = DataLoader(dataset, batch_size=4, num_workers=8)

# 2. Enable pin_memory
loader = DataLoader(dataset, batch_size=4, pin_memory=True)

# 3. Use caching
dataset = create_treno_dataset_from_csv(..., cache_dir='./cache')

# 4. Mixed precision training
# See Pattern 3 above
```

### Issue 3: NaN Loss

**Symptoms**: Loss becomes NaN during training

**Solutions**:
```python
# 1. Reduce learning rate
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)  # Lower

# 2. Use gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 3. Check data normalization
# Ensure images are normalized (mean=0, std=1)

# 4. Use different loss
criterion = nn.HuberLoss()  # More stable than MSE
```

### Issue 4: Poor Performance

**Solutions**:
```python
# 1. Use U-Net++ instead of U-Net
model = EMUNetPP(...)

# 2. Enable attention
use_attention = True

# 3. Enable radiomics
use_radiomics = True

# 4. Larger model
num_filters = [64, 128, 256, 512, 1024]
depth = 5

# 5. More data augmentation
augmentation = True
```

---

## API Reference

### Model Initialization Parameters

**All Classification/Regression/Segmentation Models** (`EMUNet`, `EMUNetPP`, `EMResNet`, `EMLeNet`, `EMUNet1D`):

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `in_channels` | int | **required** | Number of input channels |
| `out_channels` | int | **required** | Number of output channels/classes |
| `dimension` | int | **required** | 1, 2, or 3 |
| `task` | str | **required** | `'classification'`, `'regression'`, or `'segmentation'` |
| `use_attention` | bool | `False` | Enable CBAM attention |
| `use_radiomics` | bool | `False` | Extract radiomics features |
| `extra_params_dim` | int | `0` | Dimension of extra parameters |
| `num_filters` | list[int] | `[64,128,256,512]` | Filters per level |
| `depth` | int | `4` | Number of encoder levels |
| `dropout` | float | `0.0` | Dropout rate |
| `radiomics_radii` | list[int] | `[1]` | GLCM radii |
| `num_bins` | int | `64` | Histogram bins |

**Map-to-Map Models** (`EMUNetMapToMap`, `EMUNetPPMapToMap`, `UNet1DOptimized`):

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `in_channels` | int | **required** | Input image channels |
| `out_channels` | int | **required** | Output image channels |
| `dimension` | int | **required** | 1, 2, or 3 |
| `output_activation` | str | `None` | `'sigmoid'`, `'tanh'`, or `None` |
| `use_attention` | bool | `False` | Enable CBAM |
| `num_filters` | list[int] | `[64,128,256,512]` | Filters per level |
| `depth` | int | `4` | Number of levels |
| `dropout` | float | `0.0` | Dropout rate |

**1D-Specific** (`UNet1DOptimized`):

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_dilated` | bool | `True` | Use dilated convolutions |
| `dilation_schedule` | list[int] | `[1,2,4,8]` | Dilation rates |
| `pooling_type` | str | `'max'` | `'max'`, `'avg'`, `'adaptive'` |

### Forward Pass

**Classification/Regression**:
```python
output = model(x)                              # Without extra params
output = model(x, extra_params=metadata)       # With extra params
```

**Segmentation**:
```python
mask = model(x)                                # [B, C, D, H, W]
```

**Map-to-Map**:
```python
reconstructed = model(x)                       # [B, out_channels, D, H, W]
```

### Data Loading Functions

**`create_treno_dataset_from_csv`**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `csv_file` | str | **required** | Path to CSV file |
| `target_size` | list[int] | `None` | Target spatial size |
| `target_spacing` | float/list | `None` | Target spacing (mm) |
| `augmentation` | bool | `False` | Enable random flips |
| `cache_dir` | str | `None` | Directory for caching |
| `image_columns` | list[str] | `None` | Column names for images |
| `label_column` | str | `'label'` | Column name for labels |
| `labelmap_column` | str | `None` | Column name for masks |
| `custom_transforms` | Compose | `None` | Custom transform pipeline |

---

## Summary Decision Tree

```
START: What is your task?

├─ Classification
│  ├─ 3D Medical Images → EMLeNet or EMResNet (dimension=3, task='classification')
│  ├─ 2D Images → EMLeNet or EMResNet (dimension=2, task='classification')
│  └─ 1D Signals → EMUNet1D (task='classification') or EMLeNet (dimension=1)
│
├─ Regression
│  ├─ 3D/2D → EMLeNet or EMResNet (task='regression')
│  └─ 1D → EMUNet1D (task='regression') or EMLeNet (dimension=1)
│
├─ Segmentation
│  ├─ Need best quality → EMUNetPP (segmentation-only)
│  └─ Fast baseline → EMUNet (segmentation-only)
│
└─ Map-to-Map (Image Translation)
   ├─ Need best quality → EMUNetPPMapToMap
   ├─ Standard → EMUNetMapToMap
   └─ 1D sequences → UNet1DOptimized (task='maptomap')
```

---

**Ready to build? Start with the Quick Start examples and adapt to your specific use case!**

---

## Explainability (Grad-CAM & Saliency Maps)

### Quick Start

```python
from treno import GradCAM, compute_saliency_map, postprocess_cam
import torch

# Load your model
model = YourModel()
model.eval()

# Prepare input
input_tensor = torch.randn(1, 1, 64, 64, 64).requires_grad_(True)

# 1. GRAD-CAM
gradcam = GradCAM(model, target_layer=model.encoder[-1])
cam = gradcam(input_tensor, target_class=1)
cam_upsampled = gradcam.upsample_cam(cam, input_tensor.shape[-3:])

# 2. SALIENCY MAP
saliency = compute_saliency_map(model, input_tensor, target_class=1)

# 3. POST-PROCESS (smooth, normalize, mask)
cam_final = postprocess_cam(cam_upsampled, mask=brain_mask, smooth=True)
saliency_final = postprocess_cam(saliency, mask=brain_mask, smooth=True)
```

### Choosing the Right Target Layer

```python
# For U-Net style models (EMUNet)
gradcam = GradCAM(model, target_layer=model.encoder[-1])  # Last encoder layer

# For LeNet style models (EMLeNet)
gradcam = GradCAM(model, target_layer=model.features[-2])  # Before pooling

# General rule: Last convolutional layer before global pooling
```

### Best Practices

1. **Post-Processing Order**: smooth → normalize → mask
2. **If Grad-CAM is all zeros**: Check gradient flow, verify target layer
3. **Noisy saliency maps**: Apply more smoothing with `smooth_size=5` or `7`
4. **GPU acceleration**: Keep tensors on device, everything is GPU-compatible

---

## Migration from Legacy Loaders

### Old Way (Legacy)

```python
from treno.loaders import ImageLabelmapDataset

all_transforms = {'resize': [320, 320, 120], 'normalizex': 'max'}

dataset = ImageLabelmapDataset(
    annotations_file='train.csv',
    transform=all_transforms,
    index=[0, 1, 2]
)
```

### New Way (Modern)

```python
from treno.loaders import create_treno_dataset_from_csv
from torch.utils.data import DataLoader

dataset = create_treno_dataset_from_csv(
    csv_file='train.csv',
    target_size=[64, 64, 64],
    target_spacing=2.0,
    augmentation=True,
    cache_dir='./cache'
)

loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)
```

### Transform Migration

| Old Transform Dict | New pyable Transform |
|-------------------|----------------------|
| `'resize': [H, W, D]` | `target_size=[H, W, D]` in dataset |
| `'normalizex': 'max'` | `IntensityNormalization(method='max')` |
| `'normalizex': 'z'` | `IntensityNormalization(method='z-score')` |
| N/A | `RandomFlip(axes=[1, 2], prob=0.5)` |
| N/A | `RandomRotation90(axes=(1, 2), prob=0.3)` |

### Benefits of Migration

✅ **Proper pyable v3 integration** - Correct ZYX array conventions  
✅ **Automatic label preservation** - No interpolated label values  
✅ **Better performance** - Smart caching system  
✅ **More flexible** - Supports JSON, CSV, multi-CSV formats  
✅ **Modular transforms** - Composable augmentation pipeline

---

## Custom Loss Functions (treno.losses)

Treno provides specialized loss functions for medical imaging segmentation beyond standard PyTorch losses.

### Import

```python
from treno.losses import (
    EMLabelMapLoss,       # Flexible metric-based segmentation loss
    dice_loss3D,          # Dice loss for 3D volumes
    jacard_loss3D,        # Jaccard/IoU loss for 3D volumes
    EMCrossEntropyLoss,   # CrossEntropy with auto squeeze
    EMMulticlassLoss,     # Combined CE + Dice + Jaccard loss
)
```

### EMLabelMapLoss

**Best for**: Custom metric-based optimization for segmentation

```python
from treno.losses import EMLabelMapLoss

loss_fn = EMLabelMapLoss(
    num_classes=4,           # Number of segmentation classes
    logit=True,              # Input is raw logits (will argmax)
    dice=True,               # Include Dice coefficient
    jacard=True,             # Include Jaccard/IoU
    overall_accuracy=False,  # Include pixel accuracy
    label_accuracy=False,    # Include per-class accuracy
    avoid_classes=[0],       # Ignore background class (optional)
    average_loss=True        # Average all metrics
)

# Forward pass
pred = model(images)    # [B, C, D, H, W] logits
loss = loss_fn(pred, masks)  # Computes 1 - avg(metrics)
```

### Dice & Jaccard Loss (3D)

```python
from treno.losses import dice_loss3D, jacard_loss3D

dice_loss = dice_loss3D()
jaccard_loss = jacard_loss3D()

# Forward
pred = model(images)         # [B, C, D, H, W] logits
loss_d = dice_loss(pred, masks)
loss_j = jaccard_loss(pred, masks)
```

### EMMulticlassLoss

**Best for**: Combining multiple losses for robust segmentation training

```python
from treno.losses import EMMulticlassLoss
import torch

# Combines CrossEntropy + Dice + Jaccard (averaged)
loss_fn = EMMulticlassLoss(
    weight=torch.tensor([0.1, 1.0, 2.0, 3.0]),  # Class weights
    dimension=3                                   # 3D volumes
)

loss = loss_fn(pred, masks)  # Returns (CE + Dice + Jaccard) / 3
```

---

## Utility Functions (treno.utils)

### Feature Selection Pipeline

```python
from treno.utils import feature_selection

# Complete feature selection workflow
selected_features, scores = feature_selection(
    features,              # DataFrame of features
    labels,                # Target labels
    n_features=20,         # Number of features to select
    method='gini',         # 'gini', 'mutual_info', 'f_classif'
    correlation_threshold=0.9,  # Remove correlated features
    mad_filter=True        # Filter low variance features
)
```

### Medical-Aware Data Splitting

**Critical for medical imaging**: Prevents patient leakage between train/test sets.

```python
from treno.utils import stratified_group_split, extract_patient_groups

# Automatic patient group extraction (handles augmentation suffixes)
groups = extract_patient_groups(df.index, augmentation_suffix='-aug')

# Split ensuring no patient appears in both train and test
X_train, X_test, y_train, y_test, g_train, g_test = stratified_group_split(
    X, y,
    groups=None,              # Auto-extracted from index
    test_size=0.25,
    random_state=42,
    augmentation_suffix='-aug'
)
```

### Embedding Visualization

```python
from treno.utils import visualize_embeddings

# Extract features from trained model
features, labels = [], []
model.eval()
with torch.no_grad():
    for batch in dataloader:
        feat, _ = model.extract_features(batch['images'])
        features.append(feat.flatten(1).cpu().numpy())
        labels.append(batch['label'].numpy())

# Visualize with t-SNE or PCA
visualize_embeddings(features, labels, method='tsne', save_path='embeddings.png')
```

### Metrics and Evaluation

```python
from treno.utils import (
    compute_metrics,
    compute_binary_metrics,
    compute_multilabel_sensitivity_specificity,
    write_confusion_matrix_to_tensorboard
)

# Comprehensive metrics
metrics = compute_metrics(y_true, y_pred, multilabel=False)
print(f"Accuracy: {metrics['accuracy']:.3f}")
print(f"F1: {metrics['f1']:.3f}")
print(f"Confusion Matrix:\n{metrics['confusion_matrix']}")

# Binary classification metrics
binary_metrics = compute_binary_metrics(y_true, y_pred)
# Returns: accuracy, precision, recall, f1, specificity, sensitivity

# Per-class sensitivity/specificity from confusion matrix
sens, spec = compute_multilabel_sensitivity_specificity(confusion_matrix)

# TensorBoard logging
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/experiment')
write_confusion_matrix_to_tensorboard(writer, cm, min_val=0, max_val=100, tag="val/cm", epoch=10)
```

### Other Utilities

```python
from treno.utils import (
    resize_image,         # Resize ND arrays with interpolation
    store_3d_array,       # Save 3D arrays to NIfTI format
    remove_nans,          # Clean NaN values from features/labels
    zScoreFeatures,       # Z-score normalization for features
    filterFeaturesByMAD,  # Filter by Median Absolute Deviation
    filterFeaturesByCorrelation,  # Remove highly correlated features
)

# Resize image
resized = resize_image(arr, target_size=[128, 128, 128])

# Z-score features
normalized = zScoreFeatures(feature_dataframe)

# Remove correlated features (keep one from each correlated pair)
filtered = filterFeaturesByCorrelation(features, threshold=0.9)
```
