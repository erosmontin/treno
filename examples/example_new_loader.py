#!/usr/bin/env python3
"""
Example: Using the new TrenoDataset with pyable-dataloader

This example shows how to:
1. Convert a CSV file to a manifest
2. Create a TrenoDataset with transforms
3. Use it with PyTorch DataLoader
4. Train a simple model
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import tempfile
import json
import numpy as np
import SimpleITK as sitk
from pathlib import Path

# Import new treno loaders
from treno.loaders import (
    TrenoDataset,
    create_manifest_from_csv,
    save_manifest,
    create_treno_dataset_from_csv,
    PYABLE_DATALOADER_AVAILABLE
)

# Import transforms from pyable-dataloader
if PYABLE_DATALOADER_AVAILABLE:
    from pyable_dataloader import (
        Compose,
        IntensityNormalization,
        RandomFlip,
        RandomRotation90,
        RandomNoise
    )


def create_synthetic_data(output_dir, num_subjects=10):
    """Create synthetic medical images for testing."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    csv_data = []
    
    for i in range(num_subjects):
        # Create synthetic image
        arr = np.random.randn(32, 32, 32).astype(np.float32) * 100 + 500
        img = sitk.GetImageFromArray(arr)
        img.SetSpacing([2.0, 2.0, 2.0])
        img.SetOrigin([0.0, 0.0, 0.0])
        
        img_path = output_dir / f'sub{i:03d}_image.nii.gz'
        sitk.WriteImage(img, str(img_path))
        
        # Create synthetic labelmap
        labelmap = np.zeros((32, 32, 32), dtype=np.uint8)
        labelmap[10:22, 10:22, 10:22] = 1  # Box in center
        labelmap[12:20, 12:20, 12:20] = 2  # Smaller box inside
        
        lm_img = sitk.GetImageFromArray(labelmap)
        lm_img.CopyInformation(img)
        
        lm_path = output_dir / f'sub{i:03d}_labelmap.nii.gz'
        sitk.WriteImage(lm_img, str(lm_path))
        
        # Add to CSV
        label = i % 2  # Binary classification
        csv_data.append({
            'id': f'sub{i:03d}',
            'image': str(img_path),
            'labelmap': str(lm_path),
            'label': float(label)
        })
    
    # Save CSV
    import pandas as pd
    df = pd.DataFrame(csv_data)
    csv_path = output_dir / 'data.csv'
    df.to_csv(csv_path, index=False)
    
    print(f"✅ Created {num_subjects} synthetic subjects in {output_dir}")
    print(f"   CSV: {csv_path}")
    
    return csv_path


def example_1_convert_csv_to_manifest(csv_file):
    """Example 1: Convert CSV to manifest."""
    print("\n" + "="*60)
    print("Example 1: Convert CSV to Manifest")
    print("="*60)
    
    # Create manifest from CSV
    manifest = create_manifest_from_csv(
        csv_file,
        image_columns=['image'],
        labelmap_column='labelmap',
        label_column='label',
        id_column='id'
    )
    
    # Save to JSON
    manifest_path = csv_file.parent / 'manifest.json'
    save_manifest(manifest, manifest_path)
    
    print(f"✅ Created manifest with {len(manifest)} subjects")
    print(f"   Saved to: {manifest_path}")
    print(f"\n   First entry:")
    first_key = list(manifest.keys())[0]
    print(f"   {first_key}: {manifest[first_key]}")
    
    return manifest_path


def example_2_create_dataset_with_transforms(manifest_path):
    """Example 2: Create dataset with custom transforms."""
    print("\n" + "="*60)
    print("Example 2: Create Dataset with Transforms")
    print("="*60)
    
    # Define transforms
    transforms = Compose([
        IntensityNormalization(method='zscore'),
        RandomFlip(axes=[1, 2], prob=0.5),
        RandomRotation90(axes=(1, 2), prob=0.3),
        RandomNoise(std=0.01, prob=0.2)
    ])
    
    # Create dataset
    dataset = TrenoDataset(
        manifest=str(manifest_path),
        target_size=[32, 32, 32],
        target_spacing=2.0,
        transforms=transforms,
        cache_dir=str(manifest_path.parent / 'cache'),
        stack_channels=True
    )
    
    print(f"✅ Created dataset with {len(dataset)} samples")
    print(f"   Target size: [32, 32, 32]")
    print(f"   Target spacing: 2.0 mm")
    print(f"   Transforms: zscore normalization + augmentation")
    
    # Get a sample
    sample = dataset[0]
    print(f"\n   Sample 0:")
    print(f"   - images shape: {sample['images'].shape}")
    print(f"   - labelmaps shape: {sample['labelmaps'].shape}")
    print(f"   - label: {sample['label']}")
    print(f"   - id: {sample['id']}")
    
    return dataset


def example_3_use_with_dataloader(dataset):
    """Example 3: Use with PyTorch DataLoader."""
    print("\n" + "="*60)
    print("Example 3: Use with PyTorch DataLoader")
    print("="*60)
    
    # Create DataLoader
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    print(f"✅ Created DataLoader")
    print(f"   Batch size: 4")
    print(f"   Num workers: 2")
    
    # Get a batch
    batch = next(iter(loader))
    
    print(f"\n   First batch:")
    print(f"   - images shape: {batch['images'].shape}")
    print(f"   - labelmaps shape: {batch['labelmaps'].shape}")
    print(f"   - labels shape: {batch['label'].shape}")
    print(f"   - IDs: {batch['id']}")
    
    return loader


def example_4_simple_training_loop(loader):
    """Example 4: Simple training loop."""
    print("\n" + "="*60)
    print("Example 4: Simple Training Loop")
    print("="*60)
    
    # Simple 3D UNet-style model
    class SimpleSegmentationModel(nn.Module):
        def __init__(self, in_channels=1, out_channels=3):
            super().__init__()
            self.conv1 = nn.Conv3d(in_channels, 16, kernel_size=3, padding=1)
            self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
            self.conv3 = nn.Conv3d(32, out_channels, kernel_size=1)
            self.relu = nn.ReLU(inplace=True)
        
        def forward(self, x):
            x = self.relu(self.conv1(x))
            x = self.relu(self.conv2(x))
            x = self.conv3(x)
            return x
    
    # Create model
    model = SimpleSegmentationModel(in_channels=1, out_channels=3)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    print("✅ Created model and optimizer")
    
    # Training loop
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    print("\n   Training for 1 epoch...")
    for batch_idx, batch in enumerate(loader):
        images = batch['images']  # [B, C, D, H, W]
        labelmaps = batch['labelmaps']  # [B, 1, D, H, W]
        
        # Forward pass
        outputs = model(images)  # [B, 3, D, H, W]
        
        # Compute loss
        # CrossEntropyLoss expects [B, C, D, H, W] and [B, D, H, W]
        loss = criterion(outputs, labelmaps.squeeze(1).long())
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if batch_idx < 2:  # Print first 2 batches
            print(f"   Batch {batch_idx}: loss = {loss.item():.4f}")
    
    avg_loss = total_loss / num_batches
    print(f"\n✅ Training complete")
    print(f"   Average loss: {avg_loss:.4f}")


def example_5_convenience_function(csv_file):
    """Example 5: Use convenience function."""
    print("\n" + "="*60)
    print("Example 5: Convenience Function")
    print("="*60)
    
    # One-liner to create dataset from CSV
    dataset = create_treno_dataset_from_csv(
        csv_file=str(csv_file),
        target_size=[32, 32, 32],
        target_spacing=2.0,
        augmentation=True,
        cache_dir=str(csv_file.parent / 'cache'),
        image_columns=['image'],
        labelmap_column='labelmap',
        label_column='label',
        id_column='id'
    )
    
    print(f"✅ Created dataset in one line!")
    print(f"   {len(dataset)} samples")
    print(f"   With augmentation enabled")
    
    return dataset


def main():
    """Run all examples."""
    print("="*60)
    print("TrenoDataset Examples")
    print("="*60)
    
    if not PYABLE_DATALOADER_AVAILABLE:
        print("❌ pyable-dataloader is not available")
        print("Install with: pip install -e /path/to/pyable-dataloader")
        return
    
    print("✅ pyable-dataloader is available")
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create synthetic data
        csv_file = create_synthetic_data(tmpdir, num_subjects=10)
        
        # Run examples
        manifest_path = example_1_convert_csv_to_manifest(csv_file)
        dataset = example_2_create_dataset_with_transforms(manifest_path)
        loader = example_3_use_with_dataloader(dataset)
        example_4_simple_training_loop(loader)
        dataset2 = example_5_convenience_function(csv_file)
        
        print("\n" + "="*60)
        print("All examples completed successfully! 🎉")
        print("="*60)
        print("\nNext steps:")
        print("1. Replace your CSV files with real data")
        print("2. Adjust target_size and target_spacing")
        print("3. Customize transforms for your task")
        print("4. Train your model!")
        print("\nSee MIGRATION_GUIDE.md for more details.")
        print("="*60)


if __name__ == "__main__":
    main()
