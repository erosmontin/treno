"""
unet_variants.py

Canonical instantiations of:
- U-Net (Ronneberger et al. 2015)
- V-Net (Milletari et al. 2016)
- nnU-Net-style U-Net (Isensee et al.)

Built strictly by configuring existing building blocks
from model.py (no reimplementation).
"""

import torch
import torch.nn as nn

# ---------------------------------------------------------------------
# Import ONLY the required components from your model.py
# ---------------------------------------------------------------------

from models import (
    UNetBase,
    NetworkHead
)

# =====================================================================
# 1. TRUE U-NET (canonical, no residuals, no attention)
# =====================================================================

class TrueUNet(nn.Module):
    """
    Canonical U-Net:
    - Encoder–decoder
    - Skip concatenation
    - No residuals
    - No attention
    - Fully convolutional
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dimension: int = 2,
        num_filters = [64, 128, 256, 512]
    ):
        super().__init__()

        self.backbone = UNetBase(
            in_channels=in_channels,
            num_filters=num_filters,
            dimension=dimension,
            kernel_size=3,
            use_batchnorm=True,
            activation='relu',
            dropout_rate=0.0,
            bias=False,
            use_residual=False,        # ❌ residuals OFF
            use_attention=False,       # ❌ attention OFF
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
# 2. TRUE V-NET (3D, residual everywhere, no attention)
# =====================================================================

class TrueVNet(nn.Module):
    """
    V-Net-style architecture:
    - Native 3D
    - Residual blocks everywhere
    - No attention
    - Dice-oriented segmentation backbone
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
            dimension=3,               # ❗ V-Net is 3D
            kernel_size=3,
            use_batchnorm=True,
            activation='relu',
            dropout_rate=0.0,
            bias=False,
            use_residual=True,         # ✔ residuals ON
            use_attention=False,       # ❌ attention OFF
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
# 3. TRUE nnU-NET (plain U-Net, data-driven configuration)
# =====================================================================

class TrueNNUNet(nn.Module):
    """
    nnU-Net-style instantiation.

    IMPORTANT:
    - Architecture is a plain U-Net
    - Intelligence is in *configuration*, not layers
    - num_filters, depth, and dimension are expected
      to be selected by a dataset-aware planner
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
            num_filters=num_filters,   # ❗ dataset-driven
            dimension=dimension,       # ❗ 2D or 3D
            kernel_size=3,
            use_batchnorm=use_batchnorm,
            activation='leaky_relu',   # nnU-Net default
            dropout_rate=0.0,
            bias=False,
            use_residual=False,        # ❌ residuals OFF (default nnU-Net)
            use_attention=False,       # ❌ attention OFF
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
# Simple sanity check
# =====================================================================

if __name__ == "__main__":
    # 2D U-Net
    model = TrueUNet(in_channels=1, out_channels=2, dimension=2)
    x = torch.randn(1, 1, 256, 256)
    y = model(x)
    print("U-Net output:", y.shape)

    # 3D V-Net
    model = TrueVNet(in_channels=1, out_channels=3)
    x = torch.randn(1, 1, 64, 64, 64)
    y = model(x)
    print("V-Net output:", y.shape)

    # nnU-Net-style 3D
    model = TrueNNUNet(
        in_channels=1,
        out_channels=3,
        dimension=3,
        num_filters=[32, 64, 128, 256]
    )
    x = torch.randn(1, 1, 96, 96, 96)
    y = model(x)
    print("nnU-Net output:", y.shape)
