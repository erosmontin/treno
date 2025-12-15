"""
unet_variants.py

Canonical instantiations of:
- U-Net (Ronneberger et al. 2015) - Original implementation
- V-Net (Milletari et al. 2016) - Original implementation  
- Residual3DUNet (V-Net-style modernized variant)
- NNUNetStyleUNet (nnU-Net-style U-Net)

Contains both LITERAL implementations of seminal architectures and
modernized variants built from existing building blocks in model.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------
# Import ONLY the required components from your model.py
# ---------------------------------------------------------------------

from models import (
    UNetBase,
    NetworkHead,
    getNdTools
)


# =====================================================================
# LITERAL: ORIGINAL U-NET (Ronneberger et al. 2015)
# =====================================================================
# Exact reproduction of the canonical U-Net architecture.
#
# Key characteristics:
#   - Valid (unpadded) convolutions only
#   - Explicit feature map cropping before concatenation
#   - 2× (Conv → ReLU) blocks per encoder/decoder level
#   - Max-pooling for downsampling
#   - Transposed convolution for upsampling
#   - NO batch normalization
#   - NO residual connections
#   - NO attention mechanisms
#   - 2D only
#
# Reference:
#   Ronneberger, O., Fischer, P., & Brox, T. (2015).
#   U-Net: Convolutional Networks for Biomedical Image Segmentation.
#   MICCAI 2015.

class OriginalUNet(nn.Module):
    """
    Exact reproduction of the canonical U-Net (Ronneberger et al. 2015).
    
    Uses valid (unpadded) convolutions and explicit feature map cropping
    to preserve the original design's spatial information flow.
    
    Input: 2D image, arbitrary size (recommended: multiple of 32)
    Output: Segmentation map
    """
    
    def __init__(self, in_channels: int = 1, out_channels: int = 2):
        """
        Args:
            in_channels: Number of input channels (default: 1 for grayscale)
            out_channels: Number of output classes (default: 2)
        """
        super().__init__()
        
        # Encoder
        self.enc1 = self._conv_block(in_channels, 64)      # 64 filters
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc2 = self._conv_block(64, 128)              # 128 filters
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc3 = self._conv_block(128, 256)             # 256 filters
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc4 = self._conv_block(256, 512)             # 512 filters
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Bottleneck
        self.bottleneck = self._conv_block(512, 1024)      # 1024 filters
        
        # Decoder (with transposed convolutions and cropping)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = self._conv_block(1024, 512)            # 512+512 → 512
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self._conv_block(512, 256)             # 256+256 → 256
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self._conv_block(256, 128)             # 128+128 → 128
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self._conv_block(128, 64)              # 64+64 → 64
        
        # Final 1×1 convolution to map to output classes
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1, padding=0)
    
    @staticmethod
    def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
        """
        Double convolution block: Conv(3×3, valid) → ReLU → Conv(3×3, valid) → ReLU
        
        No padding, no batch norm (canonical U-Net).
        """
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=0, bias=True),
            nn.ReLU(inplace=True)
        )
    
    @staticmethod
    def _crop_and_concat(encoder_feat, decoder_feat):
        """
        Center-crop encoder feature maps to match decoder spatial dimensions,
        then concatenate along channel axis.
        
        This preserves the original U-Net's exact spatial alignment.
        """
        # encoder_feat is larger; crop it to match decoder
        h_diff = encoder_feat.size(2) - decoder_feat.size(2)
        w_diff = encoder_feat.size(3) - decoder_feat.size(3)
        
        # Remove equal amounts from each side
        h_start = h_diff // 2
        w_start = w_diff // 2
        h_end = h_start + decoder_feat.size(2)
        w_end = w_start + decoder_feat.size(3)
        
        encoder_feat = encoder_feat[:, :, h_start:h_end, w_start:w_end]
        return torch.cat([encoder_feat, decoder_feat], dim=1)
    
    def forward(self, x):
        # Encoder with skip connections
        enc1 = self.enc1(x)
        x = self.pool1(enc1)
        
        enc2 = self.enc2(x)
        x = self.pool2(enc2)
        
        enc3 = self.enc3(x)
        x = self.pool3(enc3)
        
        enc4 = self.enc4(x)
        x = self.pool4(enc4)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder with skip concatenations
        x = self.upconv4(x)
        x = self._crop_and_concat(enc4, x)
        x = self.dec4(x)
        
        x = self.upconv3(x)
        x = self._crop_and_concat(enc3, x)
        x = self.dec3(x)
        
        x = self.upconv2(x)
        x = self._crop_and_concat(enc2, x)
        x = self.dec2(x)
        
        x = self.upconv1(x)
        x = self._crop_and_concat(enc1, x)
        x = self.dec1(x)
        
        # Final output
        x = self.final_conv(x)
        return x


# =====================================================================
# LITERAL: ORIGINAL V-NET (Milletari et al. 2016)
# =====================================================================
# Exact reproduction of V-Net for volumetric medical image segmentation.
#
# Key characteristics:
#   - 3D only (volumetric data)
#   - Downsampling via strided 3D convolutions (not pooling)
#   - Upsampling via transposed 3D convolutions
#   - ADDITIVE skip connections (not concatenation)
#   - Deep residual blocks with multiple convolutions per level
#   - PReLU (Parametric ReLU) activations
#   - NO batch normalization
#   - Designed for Dice loss training
#
# Reference:
#   Milletari, F., Navab, N., & Ahmadi, S. A. (2016).
#   V-Net: Fully Convolutional Neural Networks for Volumetric Medical 
#   Image Segmentation. 3DV 2016.

class VNetResidualBlock(nn.Module):
    """
    V-Net residual block: multiple 3D convolutions with PReLU activations.
    
    Implements the building block used in V-Net's encoder/decoder paths.
    """
    
    def __init__(self, in_channels: int, out_channels: int, num_convs: int = 2):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            num_convs: Number of 3×3×3 convolutions in block
        """
        super().__init__()
        
        layers = []
        for i in range(num_convs):
            ch_in = in_channels if i == 0 else out_channels
            layers.append(nn.Conv3d(ch_in, out_channels, kernel_size=3, padding=1, bias=True))
            layers.append(nn.PReLU(out_channels))
        
        self.block = nn.Sequential(*layers)
        
        # Skip connection: 1×1×1 conv if channels don't match
        self.skip = nn.Identity()
        if in_channels != out_channels:
            self.skip = nn.Conv3d(in_channels, out_channels, kernel_size=1, padding=0, bias=True)
    
    def forward(self, x):
        return self.block(x) + self.skip(x)


class OriginalVNet(nn.Module):
    """
    Exact reproduction of V-Net (Milletari et al. 2016).
    
    Designed for 3D volumetric medical image segmentation.
    Uses additive skip connections (not concatenation) and PReLU activations.
    
    Input: 3D volume, arbitrary size (e.g., [B, 1, D, H, W])
    Output: 3D segmentation map
    
    NOTE: This architecture is designed to be trained with Dice loss,
    not cross-entropy. The architecture lacks final softmax for flexibility.
    """
    
    def __init__(self, in_channels: int = 1, out_channels: int = 2):
        """
        Args:
            in_channels: Number of input channels (default: 1 for grayscale)
            out_channels: Number of output classes (default: 2)
        """
        super().__init__()
        
        # Encoder: progressively increase filters, downsample via strided conv
        self.conv1 = VNetResidualBlock(in_channels, 16, num_convs=1)
        self.down1 = nn.Conv3d(16, 32, kernel_size=2, stride=2, padding=0, bias=True)
        self.prelu_down1 = nn.PReLU(32)
        
        self.conv2 = VNetResidualBlock(32, 32, num_convs=2)
        self.down2 = nn.Conv3d(32, 64, kernel_size=2, stride=2, padding=0, bias=True)
        self.prelu_down2 = nn.PReLU(64)
        
        self.conv3 = VNetResidualBlock(64, 64, num_convs=3)
        self.down3 = nn.Conv3d(64, 128, kernel_size=2, stride=2, padding=0, bias=True)
        self.prelu_down3 = nn.PReLU(128)
        
        # Bottleneck
        self.conv4 = VNetResidualBlock(128, 128, num_convs=3)
        
        # Decoder: progressively decrease filters, upsample via transposed conv
        self.up3 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2, padding=0, bias=True)
        self.prelu_up3 = nn.PReLU(64)
        self.conv5 = VNetResidualBlock(64, 64, num_convs=3)
        
        self.up2 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2, padding=0, bias=True)
        self.prelu_up2 = nn.PReLU(32)
        self.conv6 = VNetResidualBlock(32, 32, num_convs=2)
        
        self.up1 = nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2, padding=0, bias=True)
        self.prelu_up1 = nn.PReLU(16)
        self.conv7 = VNetResidualBlock(16, 16, num_convs=1)
        
        # Final 1×1×1 convolution to map to output classes
        self.final_conv = nn.Conv3d(16, out_channels, kernel_size=1, padding=0, bias=True)
    
    def forward(self, x):
        # Encoder with additive skip connections
        conv1_out = self.conv1(x)
        down1 = self.prelu_down1(self.down1(conv1_out))
        
        conv2_out = self.conv2(down1)
        down2 = self.prelu_down2(self.down2(conv2_out))
        
        conv3_out = self.conv3(down2)
        down3 = self.prelu_down3(self.down3(conv3_out))
        
        # Bottleneck
        conv4_out = self.conv4(down3)
        
        # Decoder with additive skip connections
        up3 = self.prelu_up3(self.up3(conv4_out))
        up3 = up3 + conv3_out  # Additive skip (not concatenation!)
        conv5_out = self.conv5(up3)
        
        up2 = self.prelu_up2(self.up2(conv5_out))
        up2 = up2 + conv2_out  # Additive skip
        conv6_out = self.conv6(up2)
        
        up1 = self.prelu_up1(self.up1(conv6_out))
        up1 = up1 + conv1_out  # Additive skip
        conv7_out = self.conv7(up1)
        
        # Final output
        output = self.final_conv(conv7_out)
        return output


# =====================================================================
# MODERNIZED: RESIDUAL 3D U-NET (formerly TrueVNet)
# =====================================================================
# V-Net-inspired architecture using modern building blocks from model.py.
#
# Architectural differences from original V-Net:
#   - Uses UNetBase which may include configurable components
#   - Designed for flexibility (batch norm, dropout, attention optional)
#   - Maintains additive skip structure but built from BaseConvBlock
#   - NOT a literal V-Net; modernized variant
#
# This class is semantically more accurate than "TrueVNet" because it is
# V-Net-INSPIRED, not the original design.

class Residual3DUNet(nn.Module):
    """
    V-Net-inspired 3D U-Net using modern building blocks.
    
    This is a MODERNIZED variant, not a literal reproduction of V-Net.
    It adds configurability (batch norm, dropout, attention) while
    maintaining the spirit of residual 3D segmentation.
    
    Key differences from original V-Net:
    - Flexible batch normalization
    - Optional attention mechanisms
    - Built from reusable BaseConvBlock components
    - Maintains native 3D support and residual structure
    
    Suitable for: 3D medical image segmentation with modern training practices
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_filters = [16, 32, 64, 128]
    ):
        super().__init__()

        self.backbone = UNetBase(
            in_channels=in_channels,
            num_filters=num_filters,
            dimension=3,               # ✔ 3D only
            kernel_size=3,
            use_batchnorm=True,
            activation='relu',
            dropout_rate=0.0,
            bias=False,
            use_residual=True,         # ✔ Residuals ON
            use_attention=False,       # ❌ Attention OFF
            reduction=0,
            use_skip_attention=False
        )

        self.head = NetworkHead(
            in_channels=num_filters[0],
            out_channels=out_channels,
            dimension=3,
            task='segmentation',
            radiomics_dim=0,
            extra_params_dim=0
        )

    def forward(self, x):
        x = self.backbone(x)
        return self.head(x)


# =====================================================================
# MODERNIZED: nnU-NET STYLE U-NET (formerly TrueNNUNet)
# =====================================================================
# nnU-Net-style instantiation using modern building blocks.
#
# Architectural note:
# - Plain U-Net architecture (no residuals, no attention by default)
# - Intelligence is in *configuration*, not architectural layers
# - num_filters, depth, and dimension driven by dataset planner
# - Supports both 2D and 3D (data-dependent)
#
# This is more semantically accurate than "TrueNNUNet" because it
# describes the STYLE/approach, not a literal implementation.

class NNUNetStyleUNet(nn.Module):
    """
    nnU-Net-style instantiation.
    
    IMPORTANT DESIGN PHILOSOPHY:
    - Architecture is a plain U-Net (no residuals by default)
    - Intelligence is in *configuration*, not architectural layers
    - num_filters, depth, and dimension are dataset-driven
    - Expected to be selected by an automated dataset analysis tool
    
    Key characteristics:
    - Flexible dimension (2D or 3D based on data)
    - Data-driven filter configuration
    - Configurable batch normalization
    - LeakyReLU activation (nnU-Net default)
    - No skip attention by default
    
    Suitable for: Automated medical image segmentation with adaptive config
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dimension: int,
        num_filters,
        use_batchnorm: bool = True
    ):
        super().__init__()

        self.backbone = UNetBase(
            in_channels=in_channels,
            num_filters=num_filters,   # ❗ Dataset-driven
            dimension=dimension,       # ❗ 2D or 3D
            kernel_size=3,
            use_batchnorm=use_batchnorm,
            activation='leaky_relu',   # nnU-Net default
            dropout_rate=0.0,
            bias=False,
            use_residual=False,        # ❌ Residuals OFF (default nnU-Net)
            use_attention=False,       # ❌ Attention OFF
            reduction=0,
            use_skip_attention=False
        )

        self.head = NetworkHead(
            in_channels=num_filters[0],
            out_channels=out_channels,
            dimension=dimension,
            task='segmentation',
            radiomics_dim=0,
            extra_params_dim=0
        )

    def forward(self, x):
        x = self.backbone(x)
        return self.head(x)


# =====================================================================
# BACKWARD COMPATIBILITY ALIASES
# =====================================================================
# These aliases ensure existing code continues to work without modification.

TrueVNet = Residual3DUNet
TrueNNUNet = NNUNetStyleUNet


# =====================================================================
# SANITY TESTS
# =====================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("LITERAL IMPLEMENTATIONS (Original Architectures)")
    print("=" * 70)
    
    # Test OriginalUNet (2D)
    print("\n[1] Testing OriginalUNet (Ronneberger et al. 2015)...")
    try:
        model = OriginalUNet(in_channels=1, out_channels=2)
        x = torch.randn(1, 1, 572, 572)  # Classic U-Net input size
        y = model(x)
        print(f"  Input shape:  {x.shape}")
        print(f"  Output shape: {y.shape}")
        assert y.shape[0] == 1 and y.shape[1] == 2, "Output shape mismatch!"
        print("  ✓ OriginalUNet PASS")
    except Exception as e:
        print(f"  ✗ OriginalUNet FAIL: {e}")
    
    # Test OriginalVNet (3D)
    print("\n[2] Testing OriginalVNet (Milletari et al. 2016)...")
    try:
        model = OriginalVNet(in_channels=1, out_channels=2)
        x = torch.randn(1, 1, 64, 64, 64)  # 3D cube
        y = model(x)
        print(f"  Input shape:  {x.shape}")
        print(f"  Output shape: {y.shape}")
        assert y.shape[0] == 1 and y.shape[1] == 2, "Output shape mismatch!"
        print("  ✓ OriginalVNet PASS")
    except Exception as e:
        print(f"  ✗ OriginalVNet FAIL: {e}")
    
    print("\n" + "=" * 70)
    print("MODERNIZED IMPLEMENTATIONS (Flexible Variants)")
    print("=" * 70)
    
    # Test Residual3DUNet (modernized V-Net)
    print("\n[3] Testing Residual3DUNet (V-Net-inspired)...")
    try:
        model = Residual3DUNet(in_channels=1, out_channels=3)
        x = torch.randn(1, 1, 64, 64, 64)
        y = model(x)
        print(f"  Input shape:  {x.shape}")
        print(f"  Output shape: {y.shape}")
        assert y.shape[0] == 1 and y.shape[1] == 3, "Output shape mismatch!"
        print("  ✓ Residual3DUNet PASS")
    except Exception as e:
        print(f"  ✗ Residual3DUNet FAIL: {e}")
    
    # Test NNUNetStyleUNet (2D and 3D)
    print("\n[4] Testing NNUNetStyleUNet (2D variant)...")
    try:
        model = NNUNetStyleUNet(
            in_channels=1,
            out_channels=3,
            dimension=2,
            num_filters=[32, 64, 128, 256]
        )
        x = torch.randn(1, 1, 256, 256)
        y = model(x)
        print(f"  Input shape:  {x.shape}")
        print(f"  Output shape: {y.shape}")
        assert y.shape[0] == 1 and y.shape[1] == 3, "Output shape mismatch!"
        print("  ✓ NNUNetStyleUNet (2D) PASS")
    except Exception as e:
        print(f"  ✗ NNUNetStyleUNet (2D) FAIL: {e}")
    
    print("\n[5] Testing NNUNetStyleUNet (3D variant)...")
    try:
        model = NNUNetStyleUNet(
            in_channels=1,
            out_channels=3,
            dimension=3,
            num_filters=[32, 64, 128, 256]
        )
        x = torch.randn(1, 1, 96, 96, 96)
        y = model(x)
        print(f"  Input shape:  {x.shape}")
        print(f"  Output shape: {y.shape}")
        assert y.shape[0] == 1 and y.shape[1] == 3, "Output shape mismatch!"
        print("  ✓ NNUNetStyleUNet (3D) PASS")
    except Exception as e:
        print(f"  ✗ NNUNetStyleUNet (3D) FAIL: {e}")
    
    print("\n" + "=" * 70)
    print("BACKWARD COMPATIBILITY ALIASES")
    print("=" * 70)
    
    # Test backward compatibility
    print("\n[6] Testing backward compatibility: TrueVNet → Residual3DUNet...")
    try:
        # Old code using TrueVNet should still work
        model = TrueVNet(in_channels=1, out_channels=3)
        x = torch.randn(1, 1, 64, 64, 64)
        y = model(x)
        print(f"  TrueVNet (alias) input:  {x.shape}")
        print(f"  TrueVNet (alias) output: {y.shape}")
        assert isinstance(model, Residual3DUNet), "Alias not working!"
        print("  ✓ TrueVNet alias PASS")
    except Exception as e:
        print(f"  ✗ TrueVNet alias FAIL: {e}")
    
    print("\n[7] Testing backward compatibility: TrueNNUNet → NNUNetStyleUNet...")
    try:
        model = TrueNNUNet(
            in_channels=1,
            out_channels=3,
            dimension=3,
            num_filters=[32, 64, 128, 256]
        )
        x = torch.randn(1, 1, 96, 96, 96)
        y = model(x)
        print(f"  TrueNNUNet (alias) input:  {x.shape}")
        print(f"  TrueNNUNet (alias) output: {y.shape}")
        assert isinstance(model, NNUNetStyleUNet), "Alias not working!"
        print("  ✓ TrueNNUNet alias PASS")
    except Exception as e:
        print(f"  ✗ TrueNNUNet alias FAIL: {e}")
    
    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETED")
    print("=" * 70)
