"""
Example: Grad-CAM and Saliency Map Generation with treno utilities

This example demonstrates how to use treno's built-in GradCAM and saliency utilities
for explainable AI in medical imaging classification tasks.
"""

import os
import torch
import numpy as np
import pandas as pd
import SimpleITK as sitk
from treno import GradCAM, compute_saliency_map, postprocess_cam
import scipy.ndimage as ndimage

# ---------------------------
# Setup
# ---------------------------
def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

set_seed(420)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# Load Your Model
# ---------------------------
# Replace this with your actual model loading code
from models import MinimalClassificationModelUnbtached

model_path = "SMALL_MODEL_UNBATCHED_with_attention/best_model.pth"
model = MinimalClassificationModelUnbtached(
    in_channels=1, 
    num_classes=3, 
    dropout=0.1,
    n_adaptive_pool=3, 
    num_channels=[16, 32, 64]
)
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval().to(device)

# ---------------------------
# Initialize Grad-CAM
# ---------------------------
# Target the last convolutional layer (adjust based on your model architecture)
try:
    target_layer = model.conv_blocks[-2].conv  # Try conv attribute first
except AttributeError:
    target_layer = model.conv_blocks[-2]  # Otherwise use the block itself

gradcam = GradCAM(model, target_layer=target_layer)

# ---------------------------
# Load Dataset
# ---------------------------
from unified_dataloader import UnifiedNiftiDataset as NiftiDataset
from unified_dataloader import overlay_on_original, create_resampled_image_sitk
import pyable_eros_montin.imaginable as ima

nifti_dataset = NiftiDataset(
    csv_file='oxford_FAI_0label.csv',
    target_size=[60, 60, 60],
    csv_roi_file='oxford_FAI_0labelROI.csv',
    target_spacing=2,
    mask_with_roi=True,
    roi_morphology='none',
    augmentation=False,
    force_reload=True,
    return_transform=True
)

# Create output directory
saliency_output_dir = 'saliency_maps_output_improved'
os.makedirs(saliency_output_dir, exist_ok=True)

# Storage for features
deep_features = []
subjects = []
predictions = []
labels_all = []

# ---------------------------
# Process Each Subject
# ---------------------------
for n in range(len(nifti_dataset)):
    try:
        # Get data
        data, label, transform_info = nifti_dataset[n]
        fn = nifti_dataset.__getitem__(n, onlyfilenames=True)
        pts = fn.iloc[1].split('/data/MYDATA/OXFORD/FAIT_2_NII/')[-1].split('/BL')[0]
        
        # Prepare input
        input_image = data.unsqueeze(0).to(device)
        target_class = int(label.item())
        
        # Get prediction
        with torch.no_grad():
            logits = model(input_image)
            pred = torch.argmax(logits, dim=1).item()
        
        print(f"[{n+1}/{len(nifti_dataset)}] {pts}: label={target_class} pred={pred}")
        
        # ---------------------------
        # 1. Compute Saliency Map (using treno utility)
        # ---------------------------
        saliency_map = compute_saliency_map(
            model, 
            input_image.clone(), 
            target_class=target_class,
            smooth=False  # We'll smooth later with postprocess
        )
        
        print(f"  Saliency raw: min={saliency_map.min():.4f}, max={saliency_map.max():.4f}")
        
        # Create brain mask (dilated version of input)
        brain_mask = ndimage.binary_dilation(data.numpy().squeeze())
        
        # Post-process saliency (smooth -> normalize -> mask)
        saliency_processed = postprocess_cam(
            saliency_map,
            mask=brain_mask,
            smooth=True,
            smooth_size=3,
            normalize=True
        )
        
        # Scale for visibility
        saliency_processed = saliency_processed * 1000
        
        print(f"  Saliency processed: min={saliency_processed.min():.4f}, max={saliency_processed.max():.4f}")
        
        # ---------------------------
        # 2. Compute Grad-CAM (using treno utility)
        # ---------------------------
        cam = gradcam(input_image.clone(), target_class=target_class, normalize=False)
        
        print(f"  Grad-CAM raw: min={cam.min():.4f}, max={cam.max():.4f}")
        
        # Upsample to input size
        cam_upsampled = gradcam.upsample_cam(cam, target_size=input_image.shape[-3:])
        
        # Post-process Grad-CAM (smooth -> normalize -> mask)
        cam_processed = postprocess_cam(
            cam_upsampled,
            mask=brain_mask,
            smooth=True,
            smooth_size=3,
            normalize=True
        )
        
        print(f"  Grad-CAM processed: min={cam_processed.min():.4f}, max={cam_processed.max():.4f}")
        
        # ---------------------------
        # 3. Save Results
        # ---------------------------
        # Load original image
        original_path = fn.iloc[1]
        IM = ima.Imaginable(original_path)
        IM.dicomOrient('LPS')
        original_array = IM.getImageAsNumpy()
        original_array_transposed = np.transpose(original_array, (2, 1, 0))
        
        original_sitk = sitk.GetImageFromArray(original_array_transposed)
        original_sitk.SetSpacing(IM.getImageSpacing())
        original_sitk.SetOrigin(IM.getImageOrigin())
        original_sitk.SetDirection(IM.getImageDirection())
        
        # Create subject directory
        subject_dir = os.path.join(saliency_output_dir, pts.replace('/', '_'))
        os.makedirs(subject_dir, exist_ok=True)
        
        # Save resampled images
        resampled_image_sitk = create_resampled_image_sitk(data.numpy().squeeze(), transform_info)
        saliency_resampled_sitk = create_resampled_image_sitk(saliency_processed, transform_info)
        gradcam_resampled_sitk = create_resampled_image_sitk(cam_processed, transform_info)
        
        # Overlay to original space
        saliency_original = overlay_on_original(original_sitk, saliency_processed, transform_info)
        gradcam_original = overlay_on_original(original_sitk, cam_processed, transform_info)
        
        # Write all files
        sitk.WriteImage(original_sitk, os.path.join(subject_dir, 'original.nii.gz'))
        sitk.WriteImage(resampled_image_sitk, os.path.join(subject_dir, 'resampled.nii.gz'))
        sitk.WriteImage(saliency_resampled_sitk, os.path.join(subject_dir, 'saliency_resampled.nii.gz'))
        sitk.WriteImage(saliency_original, os.path.join(subject_dir, 'saliency_original_space.nii.gz'))
        sitk.WriteImage(gradcam_resampled_sitk, os.path.join(subject_dir, 'gradcam_resampled.nii.gz'))
        sitk.WriteImage(gradcam_original, os.path.join(subject_dir, 'gradcam_original_space.nii.gz'))
        
        # ---------------------------
        # 4. Extract Features
        # ---------------------------
        try:
            with torch.no_grad():
                x = model.conv_blocks[:-1](input_image)
                feats = x.view(x.size(0), -1)
        except Exception:
            feats = torch.zeros(1, 100)
        
        deep_features.append(feats.cpu().numpy().flatten())
        predictions.append(pred)
        labels_all.append(target_class)
        subjects.append(pts)
        
        print(f"  ✅ Saved all maps for {pts}")
        
    except Exception as e:
        print(f"  ❌ Error for subject {n}: {e}")
        import traceback
        traceback.print_exc()
        continue

# ---------------------------
# Save Features CSV
# ---------------------------
deep_features = np.vstack(deep_features)
df = pd.DataFrame(deep_features, index=subjects)
df["label"] = labels_all
df["prediction"] = predictions
df.to_csv("deepradiomics_features_improved.csv")

print(f"\n✅ Saved {df.shape[0]} subjects × {df.shape[1]-2} features")

# Cleanup
gradcam.remove_hooks()
