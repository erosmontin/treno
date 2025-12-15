"""
example_nd_maptomap.py

Examples of image-to-image translation (map-to-map) tasks across dimensions.
"""

import torch
import torch.nn.functional as F
from treno.models import EMUNetMapToMap, EMUNetPPMapToMap
from treno.unet_1d_optimized import UNet1DOptimized

# ============================================================================
# 2D Image Denoising (Map-to-Map)
# ============================================================================

def example_2d_image_denoising():
    """2D image denoising using map-to-map."""
    print("=" * 60)
    print("2D Image Denoising Example")
    print("=" * 60)
    
    # Model for single-channel image denoising
    model = EMUNetMapToMap(
        in_channels=1,
        out_channels=1,
        dimension=2,
        num_filters=[64, 128, 256, 512],
        activation_final='sigmoid'  # Normalize output to [0, 1]
    )
    
    # Dummy data: noisy images
    noisy_images = torch.randn(8, 1, 256, 256)
    
    # Forward pass
    denoised = model(noisy_images)
    
    print(f"Input (noisy) shape: {noisy_images.shape}")
    print(f"Output (denoised) shape: {denoised.shape}")
    print(f"Output range: [{denoised.min():.4f}, {denoised.max():.4f}]")
    
    # Loss function (reconstruction)
    # In practice, clean_images would be ground truth
    clean_images = torch.randn(8, 1, 256, 256)
    loss = F.mse_loss(denoised, clean_images)
    print(f"MSE Loss: {loss.item():.6f}")
    print()


def example_2d_rgb_restoration():
    """2D RGB image restoration (artifact removal)."""
    print("=" * 60)
    print("2D RGB Image Restoration Example")
    print("=" * 60)
    
    # RGB image artifact removal
    model = EMUNetMapToMap(
        in_channels=3,  # RGB
        out_channels=3,  # RGB
        dimension=2,
        activation_final='sigmoid'
    )
    
    corrupted = torch.randn(4, 3, 512, 512)
    restored = model(corrupted)
    
    print(f"Corrupted image shape: {corrupted.shape}")
    print(f"Restored image shape: {restored.shape}")
    
    # L1 loss is often better for image reconstruction
    clean = torch.randn(4, 3, 512, 512)
    loss = F.l1_loss(restored, clean)
    print(f"L1 Loss: {loss.item():.6f}")
    print()


# ============================================================================
# 3D Medical Image Synthesis
# ============================================================================

def example_3d_medical_image_synthesis():
    """3D medical image synthesis (e.g., T2 -> synthetic T1)."""
    print("=" * 60)
    print("3D Medical Image Synthesis Example")
    print("=" * 60)
    
    # T2-weighted MRI -> Synthetic T1-weighted MRI
    model = EMUNetMapToMap(
        in_channels=1,
        out_channels=1,
        dimension=3,
        num_filters=[64, 128, 256],
        activation_final='sigmoid'
    )
    
    t2_mri = torch.randn(2, 1, 128, 128, 128)
    synthetic_t1 = model(t2_mri)
    
    print(f"Input T2 shape: {t2_mri.shape}")
    print(f"Synthetic T1 shape: {synthetic_t1.shape}")
    
    # Typically use L1 or perceptual loss
    real_t1 = torch.randn(2, 1, 128, 128, 128)
    loss = F.l1_loss(synthetic_t1, real_t1)
    print(f"L1 Loss: {loss.item():.6f}")
    print()


def example_3d_multimodal_synthesis():
    """3D multi-modal synthesis (multiple inputs -> single output)."""
    print("=" * 60)
    print("3D Multi-Modal Image Synthesis")
    print("=" * 60)
    
    # Synthesize T1 from T2 and FLAIR
    model = EMUNetMapToMap(
        in_channels=2,  # T2 + FLAIR
        out_channels=1,  # Synthetic T1
        dimension=3,
        activation_final='sigmoid'
    )
    
    t2 = torch.randn(2, 1, 64, 64, 64)
    flair = torch.randn(2, 1, 64, 64, 64)
    
    # Stack inputs
    multimodal = torch.cat([t2, flair], dim=1)  # [2, 2, 64, 64, 64]
    synthetic_t1 = model(multimodal)
    
    print(f"Input (T2 + FLAIR) shape: {multimodal.shape}")
    print(f"Output (Synthetic T1) shape: {synthetic_t1.shape}")
    print()


def example_3d_maptomap_unetpp():
    """3D map-to-map using UNet++ for better reconstruction."""
    print("=" * 60)
    print("3D Map-to-Map with UNet++ (Dense Connections)")
    print("=" * 60)
    
    # UNet++ often produces better results for dense prediction
    model = EMUNetPPMapToMap(
        in_channels=1,
        out_channels=1,
        dimension=3,
        num_filters=[32, 64, 128, 256],
        activation_final='tanh'  # Normalized to [-1, 1]
    )
    
    x = torch.randn(2, 1, 96, 96, 96)
    output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")
    print()


# ============================================================================
# 1D Signal Processing
# ============================================================================

def example_1d_ecg_denoising():
    """1D ECG signal denoising."""
    print("=" * 60)
    print("1D ECG Denoising Example")
    print("=" * 60)
    
    # ECG denoising
    model = UNet1DOptimized(
        in_channels=1,
        out_channels=1,
        depth=4,
        base_filters=32,
        task='map-to-map'
    )
    
    noisy_ecg = torch.randn(16, 1, 2048)  # 2048 samples per signal
    clean_ecg = model(noisy_ecg)
    
    print(f"Noisy ECG shape: {noisy_ecg.shape}")
    print(f"Clean ECG shape: {clean_ecg.shape}")
    
    # Loss
    target_ecg = torch.randn(16, 1, 2048)
    loss = F.mse_loss(clean_ecg, target_ecg)
    print(f"MSE Loss: {loss.item():.6f}")
    print()


def example_1d_signal_enhancement():
    """1D multi-channel signal enhancement."""
    print("=" * 60)
    print("1D Multi-Channel Signal Enhancement")
    print("=" * 60)
    
    # Enhance low-SNR sensor signal with reference
    model = UNet1DOptimized(
        in_channels=2,  # Low-SNR + High-SNR reference
        out_channels=1,  # Enhanced signal
        depth=3,
        base_filters=16,
        task='map-to-map'
    )
    
    low_snr = torch.randn(8, 1, 1024)
    reference = torch.randn(8, 1, 1024)
    
    # Concatenate channels
    inputs = torch.cat([low_snr, reference], dim=1)  # [8, 2, 1024]
    enhanced = model(inputs)
    
    print(f"Low-SNR signal shape: {low_snr.shape}")
    print(f"Reference signal shape: {reference.shape}")
    print(f"Enhanced output shape: {enhanced.shape}")
    print()


def example_1d_signal_interpolation():
    """1D signal interpolation/super-resolution preparation."""
    print("=" * 60)
    print("1D Signal Interpolation")
    print("=" * 60)
    
    # Prepare for higher-resolution signal
    model = UNet1DOptimized(
        in_channels=1,
        out_channels=1,
        depth=4,
        dilation_schedule=[1, 2, 4, 8],  # Larger receptive field
        task='map-to-map'
    )
    
    low_res = torch.randn(16, 1, 512)
    # Model outputs same resolution, can be combined with upsampling
    intermediate = model(low_res)
    
    print(f"Low-resolution input: {low_res.shape}")
    print(f"Intermediate features: {intermediate.shape}")
    print()


# ============================================================================
# Training Examples
# ============================================================================

def example_maptomap_training():
    """Example training loop for map-to-map tasks."""
    print("=" * 60)
    print("Map-to-Map Training Loop Example")
    print("=" * 60)
    
    # Model
    model = EMUNetMapToMap(
        in_channels=1,
        out_channels=1,
        dimension=2,
        activation_final='sigmoid'
    )
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Dummy data
    input_images = torch.randn(8, 1, 128, 128)
    target_images = torch.randn(8, 1, 128, 128)
    
    # Forward pass
    output = model(input_images)
    
    # Loss functions for reconstruction
    mse_loss = F.mse_loss(output, target_images)
    l1_loss = F.l1_loss(output, target_images)
    
    # Combine losses (weighted average)
    total_loss = 0.5 * mse_loss + 0.5 * l1_loss
    
    # Backward pass
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    print(f"MSE Loss: {mse_loss.item():.6f}")
    print(f"L1 Loss: {l1_loss.item():.6f}")
    print(f"Total Loss: {total_loss.item():.6f}")
    print()


def example_maptomap_with_extra_params():
    """Map-to-map with extra parameters (e.g., reconstruction strength)."""
    print("=" * 60)
    print("Map-to-Map with Extra Parameters")
    print("=" * 60)
    
    # MRI reconstruction with acceleration factor
    model = EMUNetMapToMap(
        in_channels=1,
        out_channels=1,
        dimension=2,
        activation_final='sigmoid',
        extra_params_dim=1  # Acceleration factor
    )
    
    undersampled = torch.randn(4, 1, 256, 256)
    accel_factor = torch.tensor([[4.0], [6.0], [8.0], [12.0]])  # Different acceleration factors
    
    reconstructed = model(undersampled, accel_factor)
    
    print(f"Undersampled MRI shape: {undersampled.shape}")
    print(f"Acceleration factors: {accel_factor.squeeze().tolist()}")
    print(f"Reconstructed shape: {reconstructed.shape}")
    print()


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    print("\n")
    print("=" * 60)
    print("TRENO ND MAP-TO-MAP (IMAGE TRANSLATION) EXAMPLES")
    print("=" * 60)
    print("\n")
    
    # 2D Examples
    example_2d_image_denoising()
    example_2d_rgb_restoration()
    
    # 3D Examples
    example_3d_medical_image_synthesis()
    example_3d_multimodal_synthesis()
    example_3d_maptomap_unetpp()
    
    # 1D Examples
    example_1d_ecg_denoising()
    example_1d_signal_enhancement()
    example_1d_signal_interpolation()
    
    # Training Examples
    example_maptomap_training()
    example_maptomap_with_extra_params()
    
    print("=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
