#!/usr/bin/env python3
"""
Example: Using Treno Models with the New TrenoDataset

This example demonstrates:
1. Creating datasets for different tasks
2. Using EMUNet and EMLeNet models
3. Training loops for classification, regression, and segmentation
4. Using radiomics and extra parameters
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import tempfile
import json
import numpy as np
import SimpleITK as sitk
from pathlib import Path

# Import treno components
from treno.loaders import TrenoDataset, create_manifest_from_csv, PYABLE_DATALOADER_AVAILABLE
from treno.models import EMUNet, EMLeNet
from treno.losses import EMMulticlassLoss, dice_loss3D

if not PYABLE_DATALOADER_AVAILABLE:
    print("❌ pyable-dataloader is required for this example")
    print("Install with: pip install -e /path/to/pyable-dataloader")
    exit(1)

from pyable_dataloader import Compose, IntensityNormalization, RandomFlip


def create_synthetic_data(output_dir, num_subjects=20):
    """Create synthetic medical images for testing."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    csv_data = []
    
    for i in range(num_subjects):
        # Create synthetic 3D image
        arr = np.random.randn(32, 32, 32).astype(np.float32) * 100 + 500
        img = sitk.GetImageFromArray(arr)
        img.SetSpacing([2.0, 2.0, 2.0])
        img_path = output_dir / f'sub{i:03d}_image.nii.gz'
        sitk.WriteImage(img, str(img_path))
        
        # Create synthetic labelmap (3 classes)
        labelmap = np.zeros((32, 32, 32), dtype=np.uint8)
        labelmap[10:22, 10:22, 10:22] = 1  # Class 1
        labelmap[12:20, 12:20, 12:20] = 2  # Class 2
        lm_img = sitk.GetImageFromArray(labelmap)
        lm_img.CopyInformation(img)
        lm_path = output_dir / f'sub{i:03d}_labelmap.nii.gz'
        sitk.WriteImage(lm_img, str(lm_path))
        
        # Classification label and extra params
        label = i % 2  # Binary classification
        age = 40 + np.random.randn() * 10
        tr = 100 + np.random.randn() * 20
        te = 5 + np.random.randn() * 2
        
        csv_data.append({
            'id': f'sub{i:03d}',
            'image': str(img_path),
            'labelmap': str(lm_path),
            'label': float(label),
            'age': age,
            'TR': tr,
            'TE': te
        })
    
    # Save CSV
    import pandas as pd
    df = pd.DataFrame(csv_data)
    csv_path = output_dir / 'data.csv'
    df.to_csv(csv_path, index=False)
    
    print(f"✅ Created {num_subjects} synthetic subjects")
    return csv_path


def example_1_segmentation_with_unet(data_dir):
    """Example 1: 3D Segmentation with EMUNet."""
    print("\n" + "="*60)
    print("Example 1: 3D Segmentation with EMUNet")
    print("="*60)
    
    # Create manifest
    from treno.loaders import create_manifest_from_csv
    csv_file = data_dir / 'data.csv'
    manifest = create_manifest_from_csv(
        csv_file,
        image_columns=['image'],
        labelmap_column='labelmap',
        label_column='label'
    )
    
    # Define transforms
    transforms = Compose([
        IntensityNormalization(method='zscore'),
        RandomFlip(axes=[1, 2], prob=0.5)
    ])
    
    # Create dataset
    dataset = TrenoDataset(
        manifest=manifest,
        target_size=[32, 32, 32],
        target_spacing=2.0,
        transforms=transforms,
        cache_dir=str(data_dir / 'cache')
    )
    
    # Create dataloader
    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # Create model
    model = EMUNet(
        in_channels=1,
        out_channels=3,  # 3 classes (background, class1, class2)
        dimension=3,
        num_filters=[16, 32],
        task='segmentation',
        use_radiomics=True,
        use_attention=True
    )
    
    # Loss and optimizer
    criterion = EMMulticlassLoss(weight=None, dimension=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    print(f"Model: EMUNet")
    print(f"Dataset size: {len(dataset)}")
    print(f"Task: Segmentation (3 classes)")
    
    # Training loop (1 epoch)
    model.train()
    total_loss = 0.0
    for batch_idx, batch in enumerate(loader):
        images = batch['images']  # [B, 1, D, H, W]
        labelmaps = batch['labelmaps']  # [B, 1, D, H, W]
        
        # Forward pass
        outputs = model(images)  # [B, 3, D, H, W]
        
        # Compute loss
        loss = criterion(outputs, labelmaps)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx < 2:
            print(f"   Batch {batch_idx}: loss = {loss.item():.4f}")
    
    print(f"✅ Average loss: {total_loss/len(loader):.4f}")


def example_2_classification_with_lenet(data_dir):
    """Example 2: 2D Classification with EMLeNet and extra parameters."""
    print("\n" + "="*60)
    print("Example 2: 2D Classification with EMLeNet + Extra Params")
    print("="*60)
    
    # Create manifest
    from treno.loaders import create_manifest_from_csv
    csv_file = data_dir / 'data.csv'
    
    import pandas as pd
    df = pd.read_csv(csv_file)
    
    # Create manifest with extra params
    manifest = {}
    for _, row in df.iterrows():
        manifest[row['id']] = {
            'images': [row['image']],
            'label': float(row['label']),
            'aux_data': [float(row['age']), float(row['TR']), float(row['TE'])]
        }
    
    # Create dataset (2D by taking middle slice)
    transforms = Compose([
        IntensityNormalization(method='zscore')
    ])
    
    dataset = TrenoDataset(
        manifest=manifest,
        target_size=[32, 32, 16],  # Small depth for 2D slicing
        target_spacing=2.0,
        transforms=transforms,
        return_meta=True
    )
    
    # Custom collate function to handle aux_data
    def custom_collate(batch_list):
        images = torch.stack([b['images'] for b in batch_list])
        labels = torch.stack([b['label'] for b in batch_list])
        
        # Extract middle slice for 2D (along D dimension)
        images_2d = images[:, :, images.shape[2]//2, :, :]  # [B, C, H, W]
        
        # Get aux_data if available
        aux_data = None
        if 'aux_data' in batch_list[0]:
            aux_data = torch.tensor([b['aux_data'] for b in batch_list], dtype=torch.float32)
        
        return {
            'images': images_2d,
            'labels': labels,
            'aux_data': aux_data
        }
    
    loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=custom_collate)
    
    # Create model with extra parameters
    model = EMLeNet(
        in_channels=1,
        out_channels=1,  # Binary classification
        dimension=2,
        num_filters=[16, 32, 64],
        task='classification',
        extra_params_dim=3,  # age, TR, TE
        use_radiomics=True,
        fc_layers=[128, 64]
    )
    
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    print(f"Model: EMLeNet")
    print(f"Dataset size: {len(dataset)}")
    print(f"Task: Binary classification with extra params (age, TR, TE)")
    
    # Training loop
    model.train()
    total_loss = 0.0
    for batch_idx, batch in enumerate(loader):
        images = batch['images']
        labels = batch['labels']
        aux_data = batch['aux_data']
        
        # Forward pass with extra params
        outputs = model(images, extra_params=aux_data)
        
        # Compute loss
        loss = criterion(outputs, labels.unsqueeze(1))
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx < 2:
            print(f"   Batch {batch_idx}: loss = {loss.item():.4f}")
    
    print(f"✅ Average loss: {total_loss/len(loader):.4f}")


def example_3_regression_with_radiomics(data_dir):
    """Example 3: Regression with radiomics features."""
    print("\n" + "="*60)
    print("Example 3: 3D Regression with Radiomics Features")
    print("="*60)
    
    # Create manifest
    from treno.loaders import create_manifest_from_csv
    csv_file = data_dir / 'data.csv'
    
    import pandas as pd
    df = pd.read_csv(csv_file)
    
    # Create regression targets (continuous values)
    manifest = {}
    for _, row in df.iterrows():
        manifest[row['id']] = {
            'images': [row['image']],
            'label': float(row['age'])  # Predict age from image
        }
    
    # Create dataset
    transforms = Compose([
        IntensityNormalization(method='zscore')
    ])
    
    dataset = TrenoDataset(
        manifest=manifest,
        target_size=[32, 32, 32],
        target_spacing=2.0,
        transforms=transforms
    )
    
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Create model with radiomics
    model = EMUNet(
        in_channels=1,
        out_channels=1,  # Single regression output
        dimension=3,
        num_filters=[16, 32],
        task='regression',
        use_radiomics=True,
        num_bins=128,
        radii=[1, 2, 3],  # Multiple GLCM radii
        fc_layers=[256, 128, 64]
    )
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    print(f"Model: EMUNet (Regression)")
    print(f"Dataset size: {len(dataset)}")
    print(f"Task: Age prediction from 3D images")
    print(f"Features: CNN + Radiomics (FOS + GLCM)")
    
    # Training loop
    model.train()
    total_loss = 0.0
    for batch_idx, batch in enumerate(loader):
        images = batch['images']
        labels = batch['label']
        
        # Forward pass
        outputs = model(images)
        
        # Compute loss
        loss = criterion(outputs, labels.unsqueeze(1))
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx < 2:
            print(f"   Batch {batch_idx}: loss = {loss.item():.4f}, "
                  f"pred = {outputs[0].item():.2f}, true = {labels[0].item():.2f}")
    
    print(f"✅ Average loss: {total_loss/len(loader):.4f}")


def example_4_feature_extraction(data_dir):
    """Example 4: Extract features for downstream tasks."""
    print("\n" + "="*60)
    print("Example 4: Feature Extraction")
    print("="*60)
    
    # Create simple dataset
    from treno.loaders import create_manifest_from_csv
    csv_file = data_dir / 'data.csv'
    manifest = create_manifest_from_csv(
        csv_file,
        image_columns=['image'],
        label_column='label'
    )
    
    dataset = TrenoDataset(
        manifest=manifest,
        target_size=[32, 32, 32],
        target_spacing=2.0
    )
    
    # Create model
    model = EMUNet(
        in_channels=1,
        out_channels=2,
        dimension=3,
        num_filters=[16, 32],
        task='classification',
        use_radiomics=True
    )
    
    model.eval()
    
    # Get a sample
    sample = dataset[0]
    images = sample['images'].unsqueeze(0)  # Add batch dim
    
    # Extract features
    with torch.no_grad():
        bottleneck, skip_connections, radiomics = model.extract_features(images)
    
    print(f"Extracted features:")
    print(f"  - Bottleneck shape: {bottleneck.shape}")
    print(f"  - Skip connections: {[s.shape for s in skip_connections]}")
    print(f"  - Radiomics shape: {radiomics.shape if radiomics is not None else 'None'}")
    print(f"\n✅ Features can be used for:")
    print(f"  - Transfer learning")
    print(f"  - Feature visualization")
    print(f"  - Traditional ML classifiers")


def main():
    """Run all examples."""
    print("="*60)
    print("Treno Models + DataLoader Examples")
    print("="*60)
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create synthetic data
        csv_file = create_synthetic_data(tmpdir, num_subjects=20)
        
        # Run examples
        example_1_segmentation_with_unet(tmpdir)
        example_2_classification_with_lenet(tmpdir)
        example_3_regression_with_radiomics(tmpdir)
        example_4_feature_extraction(tmpdir)
        
        print("\n" + "="*60)
        print("All examples completed successfully! 🎉")
        print("="*60)
        print("\nKey takeaways:")
        print("1. ✅ Models work seamlessly with TrenoDataset")
        print("2. ✅ Support for 1D, 2D, and 3D data")
        print("3. ✅ Multi-task: segmentation, classification, regression")
        print("4. ✅ Built-in radiomics feature extraction")
        print("5. ✅ Extra parameters (age, TR, TE, etc.) support")
        print("6. ✅ Feature extraction for transfer learning")
        print("="*60)


if __name__ == "__main__":
    main()
