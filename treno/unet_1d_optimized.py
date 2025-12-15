"""
unet_1d_optimized.py

Specialized 1D U-Net architecture optimized for time-series and signal data.

Includes:
- Dilated convolutions for larger receptive fields
- Temporal pooling strategies
- Signal-specific preprocessing options
- Optimized for sequence-to-sequence tasks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

# Handle relative imports properly
try:
    from models import (
        BaseConvBlock,
        CBAM,
        NetworkHead,
        MapToMapHead,
        getNdTools,
        calculate_fos_features,
        calculate_simple_glcm_features
    )
except ImportError:
    from .models import (
        BaseConvBlock,
        CBAM,
        NetworkHead,
        MapToMapHead,
        getNdTools,
        calculate_fos_features,
        calculate_simple_glcm_features
    )


class TemporalPooling(nn.Module):
    """Flexible temporal pooling for 1D signals."""
    
    def __init__(self, kernel_size=2, stride=2, mode='max'):
        """
        Args:
            kernel_size: Size of pooling window
            stride: Stride of pooling
            mode: 'max', 'avg', or 'adaptive'
        """
        super().__init__()
        self.mode = mode
        
        if mode == 'max':
            self.pool = nn.MaxPool1d(kernel_size, stride)
        elif mode == 'avg':
            self.pool = nn.AvgPool1d(kernel_size, stride)
        elif mode == 'adaptive':
            self.pool = nn.AdaptiveAvgPool1d(1)
        else:
            raise ValueError(f"Unknown pooling mode: {mode}")
    
    def forward(self, x):
        return self.pool(x)


class DilatedConv1dBlock(nn.Module):
    """1D convolution block with dilation for increased receptive field."""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1,
                 dropout_rate=0.0, use_batchnorm=True, activation='leaky_relu'):
        super().__init__()
        
        padding = dilation * (kernel_size - 1) // 2
        
        layers = []
        layers.append(nn.Conv1d(in_channels, out_channels, kernel_size, 
                               padding=padding, dilation=dilation, bias=not use_batchnorm))
        
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(out_channels))
        
        if activation == 'leaky_relu':
            layers.append(nn.LeakyReLU(0.1, inplace=True))
        elif activation == 'relu':
            layers.append(nn.ReLU(inplace=True))
        elif activation == 'gelu':
            layers.append(nn.GELU())
        
        if dropout_rate > 0:
            layers.append(nn.Dropout1d(dropout_rate))
        
        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.block(x)


class UNet1DOptimized(nn.Module):
    """
    Optimized 1D U-Net for time-series and signal processing.
    
    Features:
    - Dilated convolutions for large receptive fields
    - Temporal skip connections
    - Flexible pooling strategies
    - Optimized for variable-length sequences
    
    Args:
        in_channels: Number of input channels (default: 1 for univariate signals)
        out_channels: Number of output channels
        depth: Number of encoding/decoding levels
        base_filters: Number of filters at first level
        dilation_schedule: List of dilation rates per level
        pooling_mode: 'max', 'avg', or 'adaptive'
        task: 'regression', 'classification', or 'map-to-map'
        use_batchnorm: Whether to use batch normalization
        dropout_rate: Dropout probability
    
    Example:
        >>> model = UNet1DOptimized(
        ...     in_channels=1, out_channels=1, depth=4,
        ...     base_filters=32, task='map-to-map'
        ... )
        >>> x = torch.randn(8, 1, 512)  # [batch, channels, length]
        >>> output = model(x)  # [8, 1, 512]
    """
    
    def __init__(self, in_channels=1, out_channels=1, depth=4, base_filters=32,
                 dilation_schedule=None, pooling_mode='max', task='map-to-map',
                 use_batchnorm=True, dropout_rate=0.0, activation='leaky_relu'):
        super().__init__()
        
        if depth < 1:
            raise ValueError("Depth must be at least 1")
        
        self.depth = depth
        self.task = task.lower()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        if dilation_schedule is None:
            dilation_schedule = [1] * depth
        
        if len(dilation_schedule) != depth:
            raise ValueError(f"Dilation schedule length ({len(dilation_schedule)}) must match depth ({depth})")
        
        # Encoder
        self.encoders = nn.ModuleList()
        self.downsample = nn.ModuleList()
        current_filters = base_filters
        
        for i in range(depth):
            self.encoders.append(DilatedConv1dBlock(
                in_channels if i == 0 else current_filters,
                current_filters,
                kernel_size=3,
                dilation=dilation_schedule[i],
                dropout_rate=dropout_rate,
                use_batchnorm=use_batchnorm,
                activation=activation
            ))
            if i < depth - 1:
                self.downsample.append(TemporalPooling(kernel_size=2, stride=2, mode=pooling_mode))
            current_filters *= 2
        
        # Bottleneck
        self.bottleneck = DilatedConv1dBlock(
            current_filters // 2,
            current_filters // 2,
            kernel_size=3,
            dilation=dilation_schedule[-1] * 2,
            dropout_rate=dropout_rate,
            use_batchnorm=use_batchnorm,
            activation=activation
        )
        
        # Decoder
        self.decoders = nn.ModuleList()
        self.upsample = nn.ModuleList()
        
        for i in range(depth - 1, -1, -1):
            curr_filters = current_filters // 2
            next_filters = curr_filters // 2 if i > 0 else base_filters
            
            self.upsample.append(nn.ConvTranspose1d(curr_filters, next_filters, kernel_size=2, stride=2))
            
            self.decoders.append(DilatedConv1dBlock(
                curr_filters + next_filters,  # concatenated with skip connection
                next_filters,
                kernel_size=3,
                dilation=dilation_schedule[i],
                dropout_rate=dropout_rate,
                use_batchnorm=use_batchnorm,
                activation=activation
            ))
            
            current_filters = next_filters
        
        # Task-specific head
        if self.task == 'map-to-map':
            self.head = nn.Conv1d(base_filters, out_channels, kernel_size=1)
        elif self.task in ['regression', 'classification']:
            # Global average pooling + FC
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.fc = nn.Linear(base_filters, out_channels)
        else:
            raise ValueError(f"Unknown task: {self.task}")
    
    def forward(self, x):
        """
        Args:
            x: [B, C, L] where L is sequence length
        
        Returns:
            Output shape depends on task:
            - map-to-map: [B, out_channels, L]
            - regression/classification: [B, out_channels]
        """
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input [B, C, L], got shape {x.shape}")
        
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, got {x.shape[1]}")
        
        # Encoding path with skip connections
        skip_connections = []
        current = x
        
        for i in range(self.depth):
            current = self.encoders[i](current)
            skip_connections.append(current)
            
            if i < self.depth - 1:
                current = self.downsample[i](current)
        
        # Bottleneck
        current = self.bottleneck(current)
        
        # Decoding path
        for i in range(self.depth):
            current = self.upsample[i](current)
            
            # Align skip connection if needed
            skip = skip_connections[self.depth - 1 - i]
            if skip.shape[2] != current.shape[2]:
                current = F.interpolate(current, size=skip.shape[2], mode='linear', align_corners=False)
            
            # Concatenate skip connection
            current = torch.cat([current, skip], dim=1)
            current = self.decoders[i](current)
        
        # Task-specific output
        if self.task == 'map-to-map':
            return self.head(current)
        else:
            # Regression/classification
            current = self.pool(current)
            current = torch.flatten(current, 1)
            return torch.sigmoid(self.fc(current)) if self.task == 'classification' else self.fc(current)


class EMUNet1D(nn.Module):
    """
    Enhanced 1D U-Net with full feature set for time-series analysis.
    
    Supports radiomics features, extra parameters, and multiple tasks.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        task: 'regression', 'classification', or 'map-to-map'
        depth: Number of encoding/decoding levels
        base_filters: Number of filters at first level
        use_radiomics: Whether to compute radiomics features
        extra_params_dim: Dimension of extra parameters
        fc_layers: Hidden layer sizes for classification/regression
        dropout_rate: Dropout probability
        activation: Activation function name
    """
    
    def __init__(self, in_channels=1, out_channels=1, task='map-to-map', depth=4,
                 base_filters=32, use_radiomics=False, extra_params_dim=0,
                 fc_layers=[512, 256], dropout_rate=0.0, activation='leaky_relu',
                 use_batchnorm=True):
        super().__init__()
        
        self.dimension = 1
        self.task = task.lower()
        self.in_channels = in_channels
        self.use_radiomics = use_radiomics
        self.extra_params_dim = extra_params_dim
        
        # Radiomics dimension: FOS (21) + GLCM (3*1 for 1D direction = 3)
        radiomics_dim = (21 + 3) * in_channels if use_radiomics else 0
        
        # 1D U-Net backbone
        self.base = UNet1DOptimized(
            in_channels=in_channels,
            out_channels=out_channels if task == 'map-to-map' else base_filters,
            depth=depth,
            base_filters=base_filters,
            task=task,
            dropout_rate=dropout_rate,
            activation=activation,
            use_batchnorm=use_batchnorm
        )
        
        # Head for non-map-to-map tasks
        if task in ['regression', 'classification']:
            self.head = NetworkHead(
                base_filters, out_channels,
                dimension=1, task=task,
                fc_layers=fc_layers,
                dropout_rate=dropout_rate,
                activation=activation,
                radiomics_dim=radiomics_dim,
                extra_params_dim=extra_params_dim
            )
    
    def _compute_radiomics(self, x):
        """Compute radiomics features from input signal."""
        stats_features = []
        for i in range(x.shape[0]):
            channelfeatures = []
            for j in range(self.in_channels):
                fos = calculate_fos_features(x[i, j], num_bins=64)
                glcm = calculate_simple_glcm_features(x[i, j], radii=[1], dimension=1)
                combined = torch.cat((fos, glcm))
                combined = (combined - combined.mean()) / (combined.std() + 1e-6)
                channelfeatures.append(combined)
            stats_features.append(torch.cat(channelfeatures))
        return torch.stack(stats_features)
    
    def forward(self, x, extra_params=None):
        """
        Args:
            x: [B, C, L] time-series data
            extra_params: [B, extra_params_dim] optional extra features
        
        Returns:
            Output shape depends on task
        """
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input [B, C, L], got {x.shape}")
        
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, got {x.shape[1]}")
        
        radiomics_features = None
        if self.use_radiomics:
            radiomics_features = self._compute_radiomics(x)
        
        if self.task == 'map-to-map':
            return self.base(x)
        else:
            features = self.base(x)
            return self.head(features, radiomics_features, extra_params)
    
    def extract_features(self, x):
        """Extract bottleneck features for analysis."""
        return self.base.bottleneck if hasattr(self.base, 'bottleneck') else None, None
