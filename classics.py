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
"""
cnn_backbone_variants.py

Canonical CNN backbone instantiations using building blocks
defined in model.py:

- AlexNet-style encoder
- VGG-style encoder
- ResNet encoder (already implemented in model.py)

All models are encoder + NetworkHead, consistent with EM* design.
"""

import torch
import torch.nn as nn

# ---------------------------------------------------------------------
# Import required building blocks from model.py
# ---------------------------------------------------------------------

from model import (
    BaseConvBlock,
    NetworkHead,
    ResNetEncoder,
    getNdTools
)

# =====================================================================
# 1. ALEXNET (encoder-only)
# =====================================================================

class AlexNetEncoder(nn.Module):
    """
    AlexNet-style encoder (dimension-agnostic).

    Large kernels early, aggressive downsampling, no residuals,
    no skip connections.
    """

    def __init__(self, in_channels, dimension=2):
        super().__init__()
        _, _, MaxPoolNd, _, _, _ = getNdTools(dimension)

        self.features = nn.Sequential(
            BaseConvBlock(
                in_channels, 64, dimension,
                kernel_size=11, stride=4,
                use_batchnorm=False,
                activation='relu',
                use_residual=False,
                use_attention=False
            ),
            MaxPoolNd(kernel_size=3, stride=2),

            BaseConvBlock(
                64, 192, dimension,
                kernel_size=5,
                use_batchnorm=False,
                activation='relu',
                use_residual=False,
                use_attention=False
            ),
            MaxPoolNd(kernel_size=3, stride=2),

            BaseConvBlock(192, 384, dimension, activation='relu', use_attention=False),
            BaseConvBlock(384, 256, dimension, activation='relu', use_attention=False),
            BaseConvBlock(256, 256, dimension, activation='relu', use_attention=False),
            MaxPoolNd(kernel_size=3, stride=2)
        )

        self.out_channels = 256

    def forward(self, x):
        return self.features(x)


class EMAlexNet(nn.Module):
    """
    AlexNet backbone + NetworkHead.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        dimension=2,
        task='classification',
        fc_layers=[1024, 512]
    ):
        super().__init__()

        self.encoder = AlexNetEncoder(in_channels, dimension)
        self.head = NetworkHead(
            in_channels=self.encoder.out_channels,
            out_channels=out_channels,
            dimension=dimension,
            task=task,
            fc_layers=fc_layers
        )

    def forward(self, x):
        x = self.encoder(x)
        return self.head(x)


# =====================================================================
# 2. VGG (encoder-only)
# =====================================================================

class VGGEncoder(nn.Module):
    """
    VGG-style encoder using repeated BaseConvBlocks.
    """

    def __init__(self, in_channels, cfg, dimension=2):
        super().__init__()
        _, _, MaxPoolNd, _, _, _ = getNdTools(dimension)

        layers = []
        current = in_channels
        for v in cfg:
            if v == 'M':
                layers.append(MaxPoolNd(kernel_size=2, stride=2))
            else:
                layers.append(BaseConvBlock(
                    current, v, dimension,
                    kernel_size=3,
                    activation='relu',
                    use_residual=False,
                    use_attention=False
                ))
                current = v

        self.features = nn.Sequential(*layers)
        self.out_channels = current

    def forward(self, x):
        return self.features(x)


# Canonical VGG configs
VGG11 = [64, 'M', 128, 'M', 256, 256, 'M',
         512, 512, 'M', 512, 512, 'M']

VGG16 = [64, 64, 'M', 128, 128, 'M',
         256, 256, 256, 'M',
         512, 512, 512, 'M',
         512, 512, 512, 'M']


class EMVGG(nn.Module):
    """
    VGG backbone + NetworkHead.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        dimension=2,
        cfg=VGG16,
        task='classification',
        fc_layers=[1024, 512]
    ):
        super().__init__()

        self.encoder = VGGEncoder(in_channels, cfg, dimension)
        self.head = NetworkHead(
            in_channels=self.encoder.out_channels,
            out_channels=out_channels,
            dimension=dimension,
            task=task,
            fc_layers=fc_layers
        )

    def forward(self, x):
        x = self.encoder(x)
        return self.head(x)


# =====================================================================
# 3. RESNET (already implemented in model.py)
# =====================================================================

class EMResNetBackbone(nn.Module):
    """
    Thin wrapper around ResNetEncoder + NetworkHead.

    This exists only for consistency with EMAlexNet / EMVGG naming.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        dimension=2,
        layers=(2, 2, 2, 2),
        base_width=64,
        task='classification',
        fc_layers=[1024, 512],
        use_batchnorm=True,
        activation='relu'
    ):
        super().__init__()

        self.encoder = ResNetEncoder(
            in_channels=in_channels,
            dimension=dimension,
            layers=layers,
            base_width=base_width,
            use_batchnorm=use_batchnorm,
            activation=activation
        )

        self.head = NetworkHead(
            in_channels=base_width * 8,
            out_channels=out_channels,
            dimension=dimension,
            task=task,
            fc_layers=fc_layers
        )

    def forward(self, x):
        x = self.encoder(x)
        return self.head(x)


# =====================================================================
# Sanity check
# =====================================================================

if __name__ == "__main__":
    # AlexNet 2D
    x = torch.randn(1, 1, 224, 224)
    model = EMAlexNet(1, 10, dimension=2)
    print("AlexNet output:", model(x).shape)

    # VGG 2D
    model = EMVGG(1, 10, dimension=2, cfg=VGG11)
    print("VGG output:", model(x).shape)

    # ResNet 3D
    x3 = torch.randn(1, 1, 64, 64, 64)
    model = EMResNetBackbone(1, 5, dimension=3, task='classification')
    print("ResNet output:", model(x3).shape)

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
