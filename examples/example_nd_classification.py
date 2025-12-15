"""
example_nd_classification.py

Examples of classification across 1D, 2D, and 3D data using Treno models.
"""

import torch
import torch.nn as nn
from treno.models import EMUNet, EMUNetPP
from treno.unet_1d_optimized import EMUNet1D

# ============================================================================
# 2D Classification: Image Classification
# ============================================================================

def example_2d_classification():
    """Simple 2D image classification."""
    print("=" * 60)
    print("2D Image Classification Example")
    print("=" * 60)
    
    # Model for 3-channel RGB to 10 classes
    model = EMUNet(
        in_channels=3,
        out_channels=10,
        dimension=2,
        task='classification',
        use_attention=True
    )
    
    # Dummy data: batch of 8 RGB images 256x256
    x = torch.randn(8, 3, 256, 256)
    
    # Forward pass
    output = model(x)  # [8, 10]
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")
    print()


def example_2d_classification_with_radiomics():
    """2D classification with radiomics features."""
    print("=" * 60)
    print("2D Classification with Radiomics")
    print("=" * 60)
    
    # Medical image classification with radiomics
    model = EMUNet(
        in_channels=1,
        out_channels=4,  # Healthy, Disease Stage 1, 2, 3
        dimension=2,
        task='classification',
        use_radiomics=True,
        num_bins=128,
        radii=[1, 2]
    )
    
    x = torch.randn(8, 1, 512, 512)
    output = model(x)  # [8, 4]
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print()


def example_2d_classification_with_extra_params():
    """2D classification with extra clinical parameters."""
    print("=" * 60)
    print("2D Classification with Extra Parameters")
    print("=" * 60)
    
    # Medical imaging with clinical metadata
    model = EMUNet(
        in_channels=1,
        out_channels=5,
        dimension=2,
        task='classification',
        extra_params_dim=3  # age, sex, BMI
    )
    
    x = torch.randn(8, 1, 256, 256)
    extra_params = torch.randn(8, 3)  # [age, sex, BMI]
    
    output = model(x, extra_params)  # [8, 5]
    
    print(f"Image shape: {x.shape}")
    print(f"Extra params shape: {extra_params.shape}")
    print(f"Output shape: {output.shape}")
    print()


# ============================================================================
# 3D Classification: Medical Volume Classification
# ============================================================================

def example_3d_classification():
    """3D volume classification (e.g., CT/MRI diagnosis)."""
    print("=" * 60)
    print("3D Volume Classification Example")
    print("=" * 60)
    
    # 3D medical imaging classification
    model = EMUNet(
        in_channels=1,
        out_channels=3,  # Normal, Benign, Malignant
        dimension=3,
        task='classification',
        num_filters=[64, 128, 256, 512],
        use_attention=True
    )
    
    # 3D volume: [batch=4, channels=1, depth=64, height=64, width=64]
    x = torch.randn(4, 1, 64, 64, 64)
    
    output = model(x)  # [4, 3]
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print()


def example_3d_classification_with_radiomics():
    """3D classification with multi-directional radiomics."""
    print("=" * 60)
    print("3D Classification with ND Radiomics")
    print("=" * 60)
    
    # 3D with multi-modal input and radiomics
    model = EMUNet(
        in_channels=2,  # T1 and T2 MRI
        out_channels=4,
        dimension=3,
        task='classification',
        use_radiomics=True,
        num_bins=256,
        radii=[1, 2, 3],  # Multiple radii
        use_attention=True
    )
    
    x = torch.randn(2, 2, 128, 128, 128)
    output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print("Note: Radiomics now computes 3-directional GLCM (Z, Y, X axes)")
    print()


def example_3d_classification_unetpp():
    """3D classification using UNet++ (dense skip connections)."""
    print("=" * 60)
    print("3D Classification with UNet++")
    print("=" * 60)
    
    # UNet++ often provides better performance than standard UNet
    model = EMUNetPP(
        in_channels=1,
        out_channels=5,
        dimension=3,
        task='classification',
        num_filters=[32, 64, 128, 256]  # Adjust for memory
    )
    
    x = torch.randn(2, 1, 128, 128, 128)
    output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print()


# ============================================================================
# 1D Classification: Time-Series Classification
# ============================================================================

def example_1d_classification():
    """1D signal classification (e.g., ECG rhythm classification)."""
    print("=" * 60)
    print("1D Signal Classification Example")
    print("=" * 60)
    
    # ECG classification
    model = EMUNet1D(
        in_channels=1,
        out_channels=5,  # Normal, AF, PVC, Others, Noise
        task='classification',
        depth=4,
        base_filters=32
    )
    
    # Time-series: [batch=32, channels=1, length=2048]
    x = torch.randn(32, 1, 2048)
    
    output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print("Example: ECG rhythm classification from 2048 samples")
    print()


def example_1d_classification_multivariate():
    """1D multivariate time-series classification."""
    print("=" * 60)
    print("1D Multivariate Signal Classification")
    print("=" * 60)
    
    # Multi-channel sensor data classification
    model = EMUNet1D(
        in_channels=3,  # Accelerometer X, Y, Z
        out_channels=4,  # Different activities
        task='classification',
        depth=3,
        base_filters=16
    )
    
    # 3-channel signals: [batch=16, channels=3, length=512]
    x = torch.randn(16, 3, 512)
    
    output = model(x)
    
    print(f"Input shape: {x.shape} (3-axis accelerometer data)")
    print(f"Output shape: {output.shape}")
    print()


def example_1d_classification_with_radiomics():
    """1D classification with signal radiomics."""
    print("=" * 60)
    print("1D Signal Classification with Radiomics")
    print("=" * 60)
    
    model = EMUNet1D(
        in_channels=1,
        out_channels=3,
        task='classification',
        use_radiomics=True,  # Extract signal statistics
        depth=3,
        base_filters=32
    )
    
    x = torch.randn(16, 1, 1024)
    output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print("Note: Radiomics extract 24 features per 1D signal")
    print()


# ============================================================================
# Loss Functions
# ============================================================================

def example_training_loop():
    """Example training loop for classification."""
    print("=" * 60)
    print("Training Loop Example")
    print("=" * 60)
    
    # Model
    model = EMUNet(
        in_channels=1,
        out_channels=3,
        dimension=2,
        task='classification'
    )
    
    # Loss function
    criterion = nn.BCELoss()  # For sigmoid output (multi-label classification)
    # or use nn.CrossEntropyLoss() for single-label classification
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Dummy data
    x = torch.randn(8, 1, 64, 64)
    y = torch.randn(8, 3)  # Target probabilities
    
    # Forward pass
    logits = model(x)  # Already sigmoid-activated
    
    # Compute loss
    loss = criterion(logits, y)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"Loss: {loss.item():.4f}")
    print("Training step completed successfully!")
    print()


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    print("\n")
    print("=" * 60)
    print("TRENO ND CLASSIFICATION EXAMPLES")
    print("=" * 60)
    print("\n")
    
    # 2D Examples
    example_2d_classification()
    example_2d_classification_with_radiomics()
    example_2d_classification_with_extra_params()
    
    # 3D Examples
    example_3d_classification()
    example_3d_classification_with_radiomics()
    example_3d_classification_unetpp()
    
    # 1D Examples
    example_1d_classification()
    example_1d_classification_multivariate()
    example_1d_classification_with_radiomics()
    
    # Training example
    example_training_loop()
    
    print("=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
