import torch
import torch.nn as nn

from torch.utils.data import Dataset

import pandas as pd
import numpy as np
import json
from pathlib import Path

# Use pyable-dataloader for all dataset functionality
try:
    from pyable_dataloader import PyableDataset, Compose
    PYABLE_DATALOADER_AVAILABLE = True
except ImportError:
    PYABLE_DATALOADER_AVAILABLE = False
    raise ImportError(
        "pyable-dataloader is required for treno loaders. "
        "Install with: pip install -e /path/to/pyable-dataloader"
    )

# ============================================================================
# MODERN DATASET CLASSES (using pyable-dataloader)
# ============================================================================

class TrenoDataset(Dataset):
    """
    Modern dataset class using pyable-dataloader.
    
    This replaces the legacy ImageImageDataset, ImageLabelmapDataset classes
    with a unified interface based on pyable-dataloader's PyableDataset.
    
    Args:
        manifest: Path to JSON manifest or CSV file, or dict manifest
        target_size: Target image size [D, H, W] or [H, W]
        target_spacing: Target spacing in mm (float or list)
        transforms: Compose object with transforms
        roi_mask: Whether to apply ROI masking
        roi_dilation: Dilation radius for ROI in mm
        cache_dir: Directory for caching preprocessed data
        return_meta: Whether to return metadata
        stack_channels: Whether to stack multiple images as channels
        
    Example:
        >>> from treno.loaders import TrenoDataset, create_manifest_from_csv
        >>> from pyable_dataloader import Compose, IntensityNormalization, RandomFlip
        >>> 
        >>> # Convert old CSV format to manifest
        >>> manifest = create_manifest_from_csv('train.csv')
        >>> 
        >>> # Create transforms
        >>> transforms = Compose([
        ...     IntensityNormalization(method='zscore'),
        ...     RandomFlip(axes=[1, 2], prob=0.5)
        ... ])
        >>> 
        >>> # Create dataset
        >>> dataset = TrenoDataset(
        ...     manifest=manifest,
        ...     target_size=[64, 64, 64],
        ...     target_spacing=2.0,
        ...     transforms=transforms,
        ...     cache_dir='./cache'
        ... )
        >>> 
        >>> # Use with DataLoader
        >>> from torch.utils.data import DataLoader
        >>> loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)
    """
    
    def __init__(
        self,
        manifest,
        target_size=None,
        target_spacing=None,
        transforms=None,
        roi_mask=False,
        roi_dilation=None,
        cache_dir=None,
        return_meta=False,
        stack_channels=True,
        **kwargs
    ):
        if not PYABLE_DATALOADER_AVAILABLE:
            raise ImportError(
                "pyable-dataloader is required for TrenoDataset. "
                "Install with: pip install -e /path/to/pyable-dataloader"
            )
        
        self.dataset = PyableDataset(
            manifest=manifest,
            target_size=target_size,
            target_spacing=target_spacing,
            transforms=transforms,
            roi_mask=roi_mask,
            roi_dilation=roi_dilation,
            cache_dir=cache_dir,
            return_meta=return_meta,
            stack_channels=stack_channels,
            **kwargs
        )
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        """
        Returns:
            dict with keys:
                - 'images': torch.Tensor [C, D, H, W] or [C, H, W]
                - 'rois': torch.Tensor or None
                - 'labelmaps': torch.Tensor or None
                - 'label': torch.Tensor (scalar)
                - 'id': str
                - 'meta': dict (if return_meta=True)
        """
        return self.dataset[idx]
    
    def get_original_space_overlayer(self, subject_id):
        """Get function to overlay predictions back to original space."""
        return self.dataset.get_original_space_overlayer(subject_id)


def create_manifest_from_csv(
    csv_file,
    image_columns=None,
    roi_column=None,
    labelmap_column=None,
    label_column='label',
    id_column=None
):
    """
    Convert legacy CSV format to manifest dict for TrenoDataset.
    
    Args:
        csv_file: Path to CSV file
        image_columns: List of column names for images, or None to auto-detect
        roi_column: Column name for ROI, or None
        labelmap_column: Column name for labelmap, or None
        label_column: Column name for classification label
        id_column: Column name for subject ID, or None to generate
    
    Returns:
        dict: Manifest compatible with PyableDataset
        
    Example:
        >>> # For CSV with format: label,image1,image2
        >>> manifest = create_manifest_from_csv(
        ...     'train.csv',
        ...     image_columns=['image1', 'image2'],
        ...     label_column='label'
        ... )
        
        >>> # For CSV with format: id,image,roi,label
        >>> manifest = create_manifest_from_csv(
        ...     'train.csv',
        ...     image_columns=['image'],
        ...     roi_column='roi',
        ...     label_column='label',
        ...     id_column='id'
        ... )
    """
    df = pd.read_csv(csv_file)
    
    # Auto-detect image columns if not provided
    if image_columns is None:
        # Assume all columns except label, roi, labelmap, id are images
        exclude = {label_column, roi_column, labelmap_column, id_column}
        image_columns = [col for col in df.columns if col not in exclude and col]
    
    manifest = {}
    
    for idx, row in df.iterrows():
        # Generate or extract subject ID
        if id_column and id_column in df.columns:
            subject_id = str(row[id_column])
        else:
            subject_id = f"subject_{idx:04d}"
        
        # Extract image paths
        image_paths = [str(row[col]) for col in image_columns if col in df.columns and pd.notna(row[col])]
        
        # Build manifest entry
        entry = {"images": image_paths}
        
        # Add ROI if present
        if roi_column and roi_column in df.columns and pd.notna(row[roi_column]):
            entry["rois"] = [str(row[roi_column])]
        
        # Add labelmap if present
        if labelmap_column and labelmap_column in df.columns and pd.notna(row[labelmap_column]):
            entry["labelmaps"] = [str(row[labelmap_column])]
        
        # Add classification label
        if label_column and label_column in df.columns:
            entry["label"] = float(row[label_column])
        
        manifest[subject_id] = entry
    
    return manifest


def save_manifest(manifest, output_path):
    """Save manifest dict to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"✅ Saved manifest to {output_path}")


def load_manifest(manifest_path):
    """Load manifest from JSON file."""
    with open(manifest_path, 'r') as f:
        return json.load(f)


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_treno_dataset_from_csv(
    csv_file,
    target_size=[64, 64, 64],
    target_spacing=2.0,
    augmentation=True,
    cache_dir='./cache',
    **kwargs
):
    """
    Convenience function to create TrenoDataset directly from CSV file.
    
    Args:
        csv_file: Path to CSV file
        target_size: Target image size
        target_spacing: Target spacing in mm
        augmentation: Whether to apply data augmentation
        cache_dir: Cache directory
        **kwargs: Additional arguments passed to create_manifest_from_csv and TrenoDataset
    
    Returns:
        TrenoDataset instance
        
    Example:
        >>> dataset = create_treno_dataset_from_csv(
        ...     'train.csv',
        ...     target_size=[64, 64, 64],
        ...     target_spacing=2.0,
        ...     augmentation=True
        ... )
    """
    if not PYABLE_DATALOADER_AVAILABLE:
        raise ImportError(
            "pyable-dataloader is required. "
            "Install with: pip install -e /path/to/pyable-dataloader"
        )
    
    # Create manifest from CSV
    manifest = create_manifest_from_csv(csv_file, **kwargs)
    
    # Create transforms if augmentation is enabled
    transforms = None
    if augmentation:
        transforms = Compose([])
    
    # Create dataset
    return TrenoDataset(
        manifest=manifest,
        target_size=target_size,
        target_spacing=target_spacing,
        transforms=transforms,
        cache_dir=cache_dir
    )


# ============================================================================
# MAIN (for testing)
# ============================================================================

if __name__=="__main__":
    print("=" * 60)
    print("Treno Loaders Module (v3)")
    print("=" * 60)
    print(f"pyable-dataloader available: {PYABLE_DATALOADER_AVAILABLE}")
    print("=" * 60)
    
    # Example usage of modern dataset
    if PYABLE_DATALOADER_AVAILABLE:
        print("\nModern TrenoDataset example:")
        print(">>> from treno.loaders import create_treno_dataset_from_csv")
        print(">>> dataset = create_treno_dataset_from_csv('train.csv')")
        print(">>> from torch.utils.data import DataLoader")
        print(">>> loader = DataLoader(dataset, batch_size=4, shuffle=True)")
        print(">>> batch = next(iter(loader))")
        print(">>> print(batch['images'].shape)")
    
    print("\n" + "=" * 60)
    print("All legacy loaders have been removed in v3.")
    print("Use TrenoDataset and PyableDataset for all new projects.")
    print("=" * 60)
