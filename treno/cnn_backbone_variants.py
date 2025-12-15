"""
cnn_backbone_variants.py

Canonical CNN backbone instantiations using building blocks
defined in model.py.

These are ENCODER-ONLY architectures and must be paired with
a NetworkHead to define the task.

Included:
- AlexNet-style encoder
- VGG-style encoder (VGG11 / VGG16)
- ResNet encoder (wrapper around existing ResNetEncoder)

Supported tasks (via NetworkHead):
- classification
- regression

NOT intended for segmentation without a decoder.
"""

import torch
import torch.nn as nn

# ---------------------------------------------------------------------
# Import ONLY the required components from model.py
# ---------------------------------------------------------------------

from models import (
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

    Characteristics:
    - Large kernels early
    - Aggressive downsampling
    - No residuals
    - No skip connections
    """

    def __init__(self, in_channels: int, dimension: int = 2):
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
    AlexNet encoder + NetworkHead.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dimension: int = 2,
        task: str = 'classification',
        fc_layers = [1024, 512]
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

    Regular structure, no residuals, very high capacity.
    """

    def __init__(self, in_channels: int, cfg, dimension: int = 2):
        super().__init__()
        _, _, MaxPoolNd, _, _, _ = getNdTools(dimension)

        layers = []
        current = in_channels

        for v in cfg:
            if v == 'M':
                layers.append(MaxPoolNd(kernel_size=2, stride=2))
            else:
                layers.append(
                    BaseConvBlock(
                        current, v, dimension,
                        kernel_size=3,
                        activation='relu',
                        use_residual=False,
                        use_attention=False
                    )
                )
                current = v

        self.features = nn.Sequential(*layers)
        self.out_channels = current

    def forward(self, x):
        return self.features(x)


# Canonical VGG configurations
VGG11 = [
    64, 'M',
    128, 'M',
    256, 256, 'M',
    512, 512, 'M',
    512, 512, 'M'
]

VGG16 = [
    64, 64, 'M',
    128, 128, 'M',
    256, 256, 256, 'M',
    512, 512, 512, 'M',
    512, 512, 512, 'M'
]


class EMVGG(nn.Module):
    """
    VGG encoder + NetworkHead.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dimension: int = 2,
        cfg = VGG16,
        task: str = 'classification',
        fc_layers = [1024, 512]
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
# 3. RESNET (wrapper around existing ResNetEncoder)
# =====================================================================

class EMResNetBackbone(nn.Module):
    """
    ResNet encoder + NetworkHead.

    Thin wrapper for consistency with EMAlexNet / EMVGG.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dimension: int = 2,
        layers = (2, 2, 2, 2),
        base_width: int = 64,
        task: str = 'classification',
        fc_layers = [1024, 512],
        use_batchnorm: bool = True,
        activation: str = 'relu'
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
