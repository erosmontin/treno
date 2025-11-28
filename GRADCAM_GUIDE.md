# Grad-CAM and Saliency Maps in treno

This guide shows how to use `treno`'s explainability utilities for generating Grad-CAM and saliency maps.

## Quick Start

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

## Understanding the Outputs

### Grad-CAM
- **What it shows**: Which spatial regions the model focuses on for a prediction
- **How it works**: Weighted combination of final conv layer activations
- **Values**: Higher values = more important for classification
- **Best for**: Understanding **where** the model looks

### Saliency Maps
- **What it shows**: Which input pixels most affect the prediction
- **How it works**: Gradient of output w.r.t. input
- **Values**: Higher values = more sensitive pixels
- **Best for**: Understanding **which pixels** matter most

## Best Practices

### 1. Post-Processing Order Matters

```python
# ✅ CORRECT ORDER: smooth → normalize → mask
cam_processed = postprocess_cam(
    cam,
    mask=brain_mask,      # Apply last
    smooth=True,          # Smooth first
    smooth_size=3,
    normalize=True        # Normalize before masking
)

# ❌ WRONG: Normalizing after masking loses relative intensities
cam_wrong = cam / cam.max()  # Normalize first
cam_wrong[mask == 0] = 0     # Mask sets to zero
cam_wrong = cam_wrong / cam_wrong.max()  # Re-normalize amplifies noise!
```

**Why this order?**
- **Smooth first**: Reduces noise before normalization
- **Normalize second**: Preserves relative intensities across the whole volume
- **Mask last**: Applies region of interest without distorting the signal

### 2. Choosing the Right Target Layer

```python
# For U-Net style models (EMUNet)
gradcam = GradCAM(model, target_layer=model.encoder[-1])  # Last encoder layer

# For LeNet style models (EMLeNet)
gradcam = GradCAM(model, target_layer=model.features[-2])  # Before pooling

# General rule: Last convolutional layer before global pooling
```

### 3. Handling Weak Signals

```python
# If Grad-CAM shows very weak activation:

# 1. Check if gradients are flowing
print(f"Gradients captured: {gradcam.gradients is not None}")

# 2. Don't over-normalize
cam = gradcam(input_tensor, normalize=False)  # Keep raw values
cam_upsampled = gradcam.upsample_cam(cam, input_size)

# 3. Scale for visibility AFTER processing
cam_final = postprocess_cam(cam_upsampled, mask=mask, normalize=True)
cam_final = cam_final * 1000  # Scale for visualization
```

### 4. Comparing Multiple Classes

```python
# Generate CAMs for all classes
cams = {}
for class_idx in range(num_classes):
    cam = gradcam(input_tensor, target_class=class_idx)
    cam_up = gradcam.upsample_cam(cam, input_size)
    cams[class_idx] = postprocess_cam(cam_up, mask=mask)

# Compare which regions are important for each class
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, num_classes)
for i, (class_idx, cam) in enumerate(cams.items()):
    axes[i].imshow(cam[:, :, cam.shape[2]//2], cmap='hot')
    axes[i].set_title(f'Class {class_idx}')
```

## Common Issues and Solutions

### Issue 1: Grad-CAM is all zeros

```python
# Solution: Check gradient flow
model.zero_grad()
output = model(input_tensor)
score = output[0, target_class]
score.backward()

# Verify gradients exist
if gradcam.gradients is None:
    print("❌ No gradients captured - check target layer")
else:
    print(f"✅ Gradients: shape={gradcam.gradients.shape}")
```

### Issue 2: Saliency map is noisy

```python
# Solution: Apply more smoothing
saliency = compute_saliency_map(model, input_tensor, smooth=True, smooth_size=5)

# Or post-process with larger kernel
saliency_smooth = postprocess_cam(saliency, smooth=True, smooth_size=7)
```

### Issue 3: CAM doesn't match prediction regions

```python
# This might be correct! The model may use:
# 1. Global features (texture, not location)
# 2. Absence of features (what's NOT there)
# 3. Multiple distributed regions

# To debug:
# 1. Visualize multiple slices (not just center)
# 2. Check if model architecture has attention mechanisms
# 3. Try integrated gradients instead of vanilla gradients
```

## Advanced: Custom Post-Processing

```python
def custom_postprocess(cam, mask=None, percentile_clip=95):
    """
    Custom post-processing with percentile-based normalization.
    """
    import scipy.ndimage as ndimage
    
    # 1. Smooth
    cam_smooth = ndimage.uniform_filter(cam, size=3)
    
    # 2. Clip outliers using percentile
    upper = np.percentile(cam_smooth, percentile_clip)
    cam_clipped = np.clip(cam_smooth, 0, upper)
    
    # 3. Normalize
    if cam_clipped.max() > 0:
        cam_norm = cam_clipped / cam_clipped.max()
    else:
        cam_norm = cam_clipped
    
    # 4. Apply mask
    if mask is not None:
        cam_norm = cam_norm * mask
    
    return cam_norm

# Use custom processing
cam_custom = custom_postprocess(cam_upsampled, mask=brain_mask, percentile_clip=98)
```

## Integration with TensorBoard

```python
from torch.utils.tensorboard import SummaryWriter
from treno import write_confusion_matrix_to_tensorboard

writer = SummaryWriter('runs/explainability')

# Log Grad-CAM as image
cam_tensor = torch.from_numpy(cam_final).unsqueeze(0)  # Add channel dim
writer.add_image('GradCAM/class_1', cam_tensor, epoch, dataformats='CHW')

# Log saliency map
saliency_tensor = torch.from_numpy(saliency_final).unsqueeze(0)
writer.add_image('Saliency/class_1', saliency_tensor, epoch, dataformats='CHW')

writer.close()
```

## Performance Tips

```python
# 1. Batch processing (if memory allows)
# Note: GradCAM works on single samples, so loop efficiently

import torch.no_grad()

predictions = []
with torch.no_grad():
    for batch in dataloader:
        preds = model(batch)
        predictions.append(preds)

# Then compute Grad-CAM only for interesting samples
for idx in interesting_indices:
    cam = gradcam(data[idx:idx+1], target_class=pred[idx])

# 2. Cache preprocessed data
# Use your dataloader's caching (force_reload=False)

# 3. GPU acceleration
# Everything is already GPU-accelerated! Just keep tensors on device.
```

## Full Workflow Example

See `examples/example_gradcam_saliency.py` for a complete working example that:
1. Loads a trained model
2. Processes a dataset
3. Generates both Grad-CAM and saliency maps
4. Saves results in both resampled and original image space
5. Extracts deep features for analysis

## References

- **Grad-CAM Paper**: Selvaraju et al. "Grad-CAM: Visual Explanations from Deep Networks" (2017)
- **Saliency Maps**: Simonyan et al. "Deep Inside Convolutional Networks" (2014)
