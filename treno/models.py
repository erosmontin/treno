import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path

def getNdTools(dimension):
    """Returns appropriate PyTorch modules based on the specified dimension."""
    if dimension not in [1, 2, 3]:
        raise ValueError("Only 1, 2, or 3 dimensions are supported")
    return {
        1: (nn.Conv1d, nn.ConvTranspose1d, nn.MaxPool1d, nn.BatchNorm1d, nn.Dropout, nn.ReflectionPad1d),
        2: (nn.Conv2d, nn.ConvTranspose2d, nn.MaxPool2d, nn.BatchNorm2d, nn.Dropout2d, nn.ReflectionPad2d),
        3: (nn.Conv3d, nn.ConvTranspose3d, nn.MaxPool3d, nn.BatchNorm3d, nn.Dropout3d, nn.ReflectionPad3d)
    }[dimension]

def calculate_skewness_torch(x):
    mean = torch.mean(x)
    std_dev = torch.std(x)
    skewness = torch.mean((x - mean) ** 3) / (std_dev ** 3 + 1e-6)
    return skewness

def calculate_kurtosis_torch(x):
    mean = torch.mean(x)
    std_dev = torch.std(x)
    kurtosis = torch.mean((x - mean) ** 4) / (std_dev ** 4 + 1e-6) - 3
    return kurtosis

def calculate_fos_features(x, num_bins=256):
    """Calculate extended first-order statistical features from a tensor."""
    x_min, x_max = torch.min(x), torch.max(x)
    x_norm = (x - x_min) / (x_max - x_min + 1e-6) if x_max > x_min else x

    energy = torch.sum(x**2)
    total_energy = torch.sum(x)
    # Move x_norm to CPU for histogram computation and move back to original device
    h = torch.histogram(x_norm.flatten().cpu(), bins=num_bins, density=True)[0].to(x_norm.device)
    h = h[h > 1e-5]
    entropy = -torch.sum(h * torch.log(h + 1e-6))
    minimum = torch.min(x)
    percentiles = torch.quantile(x.flatten(), torch.tensor([0.1, 0.25, 0.75, 0.9], device=x.device))
    maximum = torch.max(x)
    mean = torch.mean(x)
    median = torch.median(x)
    mode = h.argmax() / (num_bins - 1)
    interquartile_range = percentiles[2] - percentiles[1]
    range_ = maximum - minimum
    mad = torch.mean(torch.abs(x - mean))
    mad_median = torch.mean(torch.abs(x - median))
    rms = torch.sqrt(torch.mean(x**2))
    std_dev = torch.std(x)
    variance = torch.var(x)
    skewness = calculate_skewness_torch(x)
    kurtosis = calculate_kurtosis_torch(x)
    unique_elements = torch.unique(x)
    uniformity = len(unique_elements) / x.numel()
    cv = std_dev / (mean + 1e-6)
    diff_entropy = -torch.sum(torch.diff(x.flatten()) * torch.log(torch.abs(torch.diff(x.flatten())) + 1e-6))

    # Create the tensor on the same device as x
    return torch.tensor([
        energy, cv, total_energy, entropy, minimum, *percentiles, maximum, mean,
        median, mode, interquartile_range, range_, mad, mad_median, rms, std_dev,
        skewness, kurtosis, variance, uniformity, diff_entropy
    ], device=x.device)

def calculate_simple_glcm_features(x, radii=[1], dimension=2):
    """Calculate simplified GLCM-like features for a tensor across multiple radii and directions."""
    glcm_features = []
    
    # Define axes to shift based on dimension
    if dimension == 1:
        axes = [0]
    elif dimension == 2:
        axes = [0, 1]  # vertical and horizontal
    else:  # 3D
        axes = [0, 1, 2]  # depth, height, width
    
    for radius in radii:
        for axis in axes:
            x_shift = torch.roll(x, shifts=radius, dims=axis)
            # Zero out the wrapped-around region
            if axis == 0:
                x_shift[:radius] = 0
            elif axis == 1:
                x_shift[:, :radius] = 0
            else:  # axis == 2
                x_shift[:, :, :radius] = 0
            
            contrast = torch.mean((x - x_shift) ** 2)
            energy = torch.sum(x**2)
            homogeneity = torch.mean(1 / (1 + torch.abs(x - x_shift)))
            glcm_features.extend([contrast, energy, homogeneity])
        
    # Create the tensor on the same device as x
    return torch.tensor(glcm_features, device=x.device)

class ChannelAttention(nn.Module):
    """Channel attention module for CBAM."""
    def __init__(self, in_channels, dimension=2, reduction=16):
        super().__init__()
        self.dimension = dimension
        self.avg_pool = {
            1: nn.AdaptiveAvgPool1d(1),
            2: nn.AdaptiveAvgPool2d(1),
            3: nn.AdaptiveAvgPool3d(1)
        }[dimension]
        reduced_channels = max(in_channels // reduction, 1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels,reduced_channels , bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c = x.size(0), x.size(1)
        avg = self.avg_pool(x).view(b, c)
        fc_out = self.fc(avg)
        return self.sigmoid(fc_out.view(b, c, *[1]*self.dimension))

class SpatialAttention(nn.Module):
    """Spatial attention module for CBAM."""
    def __init__(self, dimension=2, kernel_size=7):
        super().__init__()
        if dimension < 2:
            self.conv = nn.Identity()
        else:
            padding = (kernel_size - 1) // 2
            ConvNd, _, _, _, _, _ = getNdTools(dimension)
            self.conv = ConvNd(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if isinstance(self.conv, nn.Identity):
            return torch.ones_like(x)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x))

class CBAM(nn.Module):
    """Convolutional Block Attention Module."""
    def __init__(self, in_channels, dimension=2, reduction=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(in_channels, dimension, reduction)
        self.sa = SpatialAttention(dimension, kernel_size)
        
    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

class BaseConvBlock(nn.Module):
    """Basic convolutional block with optional attention and residual connections."""
    def __init__(self, in_channels, out_channels, dimension=2, kernel_size=3, stride=1,
                 use_batchnorm=True, activation='leaky_relu', dropout_rate=0.0,
                 leaky_slope=0.1, bias=False, use_residual=False, use_attention=True,reduction=16):
        super().__init__()
        
        ConvNd, _, _, BatchNormNd, DropoutNd, ReflectionPadNd = getNdTools(dimension)
        self.use_residual = use_residual and (in_channels == out_channels) and (stride == 1)
        padding = (kernel_size - 1) // 2
        
        layers = []
        if padding > 0:
            layers.append(ReflectionPadNd(padding))
            
        layers.append(ConvNd(in_channels, out_channels, kernel_size, stride, padding=0, bias=bias))
        
        if use_batchnorm:
            layers.append(BatchNormNd(out_channels))
            
        activation_dict = {
            'leaky_relu': nn.LeakyReLU(leaky_slope, inplace=True),
            'relu': nn.ReLU(inplace=True),
            'gelu': nn.GELU(),
            'none': nn.Identity()
        }
        self.act = activation_dict.get(activation, nn.Identity())
        layers.append(self.act)
        
        if dropout_rate > 0:
            layers.append(DropoutNd(dropout_rate))
            
        if use_attention:
            layers.append(CBAM(out_channels, dimension, reduction))
            
        self.block = nn.Sequential(*layers)
        
    def forward(self, x):
        out = self.block(x)
        if self.use_residual:
            out = out + x
        return out

class UNetBase(nn.Module):
    """Base U-Net architecture with flexible configuration."""
    def __init__(self, in_channels, num_filters=[64, 128, 256, 512], dimension=2,
                 kernel_size=3, use_batchnorm=True, activation='leaky_relu',
                 dropout_rate=0.0, leaky_slope=0.1, bias=False, use_residual=False,
                 use_attention=True,reduction=2, use_skip_attention=False):
        
        super().__init__()
        
        ConvNd, ConvTransposeNd, MaxPoolNd, _, _, _ = getNdTools(dimension)
        self.num_filters = num_filters
        self.pool = MaxPoolNd(kernel_size=2, stride=2)
        
        self.downs = nn.ModuleList()
        current_channels = in_channels
        for filters in num_filters:
            self.downs.append(BaseConvBlock(
                current_channels, filters, dimension, kernel_size, 1,
                use_batchnorm, activation, dropout_rate, leaky_slope, bias, 
                use_residual, use_attention, reduction
            ))
            current_channels = filters
            
        self.bottleneck = BaseConvBlock(
            num_filters[-1], num_filters[-1]*2, dimension, kernel_size, 1,
            use_batchnorm, activation, dropout_rate, leaky_slope, bias, 
            use_residual, use_attention,reduction
        )
        
        self.ups = nn.ModuleList()
        self.use_skip_attention = use_skip_attention
        self.skip_gates = nn.ModuleList() if use_skip_attention else None
        for filters in reversed(num_filters):
            self.ups.append(nn.Sequential(
                ConvTransposeNd(filters*2, filters, kernel_size=2, stride=2),
                BaseConvBlock(filters*2, filters, dimension, kernel_size, 1,
                            use_batchnorm, activation, dropout_rate, leaky_slope, 
                            bias, use_residual, use_attention,reduction)
            ))
            if use_skip_attention:
                # gate expects skip (filters) and decoder g (filters)
                self.skip_gates.append(AttentionGate(filters, filters, filters//2, dimension))
            
    def forward_features(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        return x, skip_connections
    
    def forward(self, x):
        x, skip_connections = self.forward_features(x)
        x = self.bottleneck(x)
        
        skip_connections = skip_connections[::-1]
        for i, up in enumerate(self.ups):
            x = up[0](x)
            skip = skip_connections[i]
            if x.shape[2:] != skip.shape[2:]:
                mode = 'linear' if len(x.shape) == 3 else 'bilinear' if len(x.shape) == 4 else 'trilinear'
                x = F.interpolate(x, size=skip.shape[2:], mode=mode, align_corners=False)
            if self.use_skip_attention:
                skip = self.skip_gates[i](skip, x)
            x = torch.cat([skip, x], dim=1)
            x = up[1](x)
        return x

class UNetPPBase(nn.Module):
    """UNet++ base with nested dense skip connections. Supports 1D/2D/3D."""
    def __init__(self, in_channels, num_filters=[64, 128, 256], dimension=2,
                 kernel_size=3, use_batchnorm=True, activation='leaky_relu',
                 dropout_rate=0.0, leaky_slope=0.1, bias=False, use_attention=True, reduction=2):
        super().__init__()
        _, ConvTransposeNd, MaxPoolNd, _, _, _ = getNdTools(dimension)
        self.pool = MaxPoolNd(kernel_size=2, stride=2)
        self.num_filters = num_filters
        
        # Encoder path
        self.enc = nn.ModuleList()
        current_channels = in_channels
        for f in num_filters:
            self.enc.append(BaseConvBlock(current_channels, f, dimension, kernel_size, 1,
                                          use_batchnorm, activation, dropout_rate, leaky_slope,
                                          bias, False, use_attention, reduction))
            current_channels = f
        
        # Nested decoder nodes X^{i,j}
        # Maintain per-level refined nodes and compute accurate concat channels
        self.nodes = nn.ModuleDict()
        for i in range(len(num_filters)):
            for j in range(1, len(num_filters)-i):
                # Inputs: X^{i,0} (enc) + upsample(X^{i+1,j-1}) + (j-1) refined nodes X^{i,1..j-1}
                # Upsample output matches num_filters[i], so: j*num_filters[i] + num_filters[i] (from deeper)
                # Actually: X[i][0..j-1] all have num_filters[i] channels, up has num_filters[i] after upsample
                in_ch = num_filters[i] * (j + 1)  # j skips from level i + 1 upsampled
                out_ch = num_filters[i]
                self.nodes[f"{i}_{j}"] = BaseConvBlock(
                    in_ch, out_ch, dimension, kernel_size, 1,
                    use_batchnorm, activation, dropout_rate, leaky_slope,
                    bias, False, use_attention, reduction
                )
        
        # Upsamplers for connecting deeper features upwards
        self.ups = nn.ModuleDict()
        for i in range(1, len(num_filters)):
            self.ups[str(i)] = ConvTransposeNd(num_filters[i], num_filters[i-1], kernel_size=2, stride=2)
        
    def forward_features(self, x):
        # Encoder outputs (X^{i,0}) stored in grid
        N = len(self.num_filters)
        X = [[None for _ in range(N)] for _ in range(N)]
        cur = x
        for i, enc in enumerate(self.enc):
            cur = enc(cur)
            X[i][0] = cur
            if i < N-1:
                cur = self.pool(cur)
        # Nested decoding
        for j in range(1, N):
            for i in range(N - j):
                up = self.ups[str(i+1)](X[i+1][j-1])
                # Align spatial size with X[i][0]
                if up.shape[2:] != X[i][0].shape[2:]:
                    mode = 'linear' if self.dimension == 1 else 'bilinear' if self.dimension == 2 else 'trilinear'
                    up = F.interpolate(up, size=X[i][0].shape[2:], mode=mode, align_corners=False)
                # Dense concat: X^{i,0} .. X^{i, j-1} plus up
                concat = [X[i][k] for k in range(0, j)] + [up]
                X[i][j] = self.nodes[f"{i}_{j}"](torch.cat(concat, dim=1))
        return X[0][N-1]
    
    def forward(self, x):
        return self.forward_features(x)

class EMUNetPP(nn.Module):
    """UNet++ for semantic segmentation with dense skip connections.
    
    This model is specifically designed for pixel/voxel-level segmentation tasks.
    For classification or regression, use EMLeNet or EMResNet instead.
    """
    def __init__(self, in_channels, out_channels, dimension=2, num_filters=[64,128,256],
                 use_batchnorm=True, activation='leaky_relu',
                 dropout_rate=0.0, leaky_slope=0.1, bias=False,
                 extra_params_dim=0, use_attention=True, use_radiomics=False,
                 num_bins=256, radii=[1]):
        super().__init__()
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("Channels must be positive")
        self.dimension = dimension
        self.in_channels = in_channels
        self.task = 'segmentation'  # Fixed task
        self.use_radiomics = use_radiomics
        self.num_bins = num_bins
        self.radii = radii
        self.extra_params_dim = extra_params_dim
        
        self.base = UNetPPBase(in_channels, num_filters, dimension, 3, use_batchnorm,
                               activation, dropout_rate, leaky_slope, bias, use_attention)
        # Radiomics: 24 FOS + (3 features × directions × radii)
        # directions: 1D=1, 2D=2, 3D=3
        num_directions = dimension
        radiomics_dim = (24 + 3 * num_directions * len(radii)) * in_channels if use_radiomics else 0
        # Use first level filters for head input similar to EMUNet
        self.head = NetworkHead(num_filters[0], out_channels, dimension, 'segmentation', [],
                                dropout_rate, activation, leaky_slope, bias, radiomics_dim, extra_params_dim)
        
        # Optional fusion: gate UNet++ output feature maps with extra_params
        self.use_fusion = extra_params_dim > 0
        if self.use_fusion:
            self.fusion = FusionHead(num_filters[0], extra_params_dim)
    
    def _compute_radiomics(self, x):
        stats_features = []
        for i in range(x.shape[0]):
            channelfeatures = []
            for j in range(self.in_channels):
                fos = calculate_fos_features(x[i, j], num_bins=self.num_bins)
                glcm = calculate_simple_glcm_features(x[i, j], radii=self.radii, dimension=self.dimension)
                combined = torch.cat((fos, glcm))
                combined = (combined - combined.mean()) / (combined.std() + 1e-6)
                channelfeatures.append(combined)
            stats_features.append(torch.cat(channelfeatures))
        return torch.stack(stats_features)
    
    def forward(self, x, extra_params=None):
        expected_dims = self.dimension + 2
        if x.dim() != expected_dims:
            raise ValueError(f"Expected {self.dimension}D input with shape [B, C, ...], got {x.shape}")
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, got {x.shape[1]}")
        radiomics_features = None
        if self.use_radiomics:
            radiomics_features = self._compute_radiomics(x)
        x = self.base(x)
        if extra_params is not None and self.extra_params_dim > 0:
            if extra_params.shape[1] != self.extra_params_dim:
                raise ValueError(f"Provided extra_params dim ({extra_params.shape[1]}) does not match initialized extra_params_dim ({self.extra_params_dim})")
        # Apply fusion gating to UNet++ output feature maps if requested
        if extra_params is not None and self.use_fusion:
            x = self.fusion(x, extra_params)
        return self.head(x, radiomics_features, extra_params)

    def extract_features(self, x):
        """Extract deep features prior to the head along with engineered radiomics.

        Returns a tuple (features, radiomics_features) where:
        - features: tensor [B, C, ...] from UNet++ output prior to NetworkHead
        - radiomics_features: tensor [B, R] or None
        """
        features = self.base.forward_features(x)
        radiomics_features = self._compute_radiomics(x) if self.use_radiomics else None
        return features, radiomics_features

class LeNetBase(nn.Module):
    """Base LeNet architecture with flexible configuration."""
    def __init__(self, in_channels, num_filters=[16, 32, 64], dimension=2,
                 kernel_size=3, use_batchnorm=True, activation='leaky_relu',
                 dropout_rate=0.0, leaky_slope=0.1, bias=False, use_residual=False,
                 use_attention=True,reduction=2):
        """LeNet base architecture with configurable parameters."""
        if len(num_filters) < 2:
            raise ValueError("LeNet requires at least 2 filter sizes")
        super().__init__()
        
        _, _, MaxPoolNd, _, _, _ = getNdTools(dimension)
        self.num_filters = num_filters
        self.pool = MaxPoolNd(kernel_size=2, stride=2)
        
        self.convs = nn.ModuleList()
        current_channels = in_channels
        for filters in num_filters[:-1]:
            self.convs.append(BaseConvBlock(
                current_channels, filters, dimension, kernel_size, 1,
                use_batchnorm, activation, dropout_rate, leaky_slope, bias, 
                use_residual, use_attention,reduction
            ))
            current_channels = filters
            
    def forward_features(self, x):
        for conv in self.convs:
            x = conv(x)
            x = self.pool(x)
        return x
    
    def forward(self, x):
        return self.forward_features(x)

class NetworkHead(nn.Module):
    """
    Configurable network head for different tasks with optional radiomics and extra parameters.
    
    Output behavior by task:
        - 'segmentation': Returns raw logits (no activation) for use with CrossEntropyLoss
        - 'classification': Returns sigmoid-activated probabilities for binary/multi-label tasks
        - 'regression': Returns raw outputs (no activation)
    
    Args:
        in_channels: Number of input channels from backbone
        out_channels: Number of output channels/classes
        dimension: Spatial dimension (1, 2, or 3)
        task: One of 'regression', 'classification', or 'segmentation'
        fc_layers: List of hidden layer sizes for regression/classification
        dropout_rate: Dropout probability
        activation: Activation function name
        leaky_slope: Negative slope for LeakyReLU
        bias: Whether to use bias in conv/linear layers
        radiomics_dim: Dimension of radiomics features (0 to disable)
        extra_params_dim: Dimension of extra parameters like age, TR, TE (0 to disable)
    """
    def __init__(self, in_channels, out_channels, dimension=2, task='regression',
                 fc_layers=[1024, 512], dropout_rate=0.0, activation='leaky_relu',
                 leaky_slope=0.1, bias=False, radiomics_dim=0, extra_params_dim=0):
        super().__init__()
        
        ConvNd, _, _, _, _, _ = getNdTools(dimension)
        self.task = task.lower()
        self.dimension = dimension
        self.radiomics_dim = radiomics_dim
        self.extra_params_dim = extra_params_dim
        
        if self.task in ['regression', 'classification']:
            self.pool = {
                1: nn.AdaptiveAvgPool1d(1),
                2: nn.AdaptiveAvgPool2d(1),
                3: nn.AdaptiveAvgPool3d(1)
            }[dimension]
            total_in_features = in_channels + radiomics_dim + extra_params_dim
            self.fc_layers = nn.ModuleList()
            current_channels = total_in_features
            for fc_size in fc_layers:
                self.fc_layers.append(nn.Sequential(
                    nn.Linear(current_channels, fc_size, bias=bias),
                    self._get_activation(activation, leaky_slope),
                    nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
                ))
                current_channels = fc_size
            self.fc_layers.append(nn.Linear(current_channels, out_channels, bias=bias))
        else:  # segmentation
            self.head = ConvNd(in_channels + radiomics_dim + extra_params_dim, out_channels, kernel_size=1, bias=bias)
            
    def _get_activation(self, activation, leaky_slope):
        activation_dict = {
            'leaky_relu': nn.LeakyReLU(leaky_slope, inplace=True),
            'relu': nn.ReLU(inplace=True),
            'gelu': nn.GELU(),
            'none': nn.Identity()
        }
        return activation_dict.get(activation, nn.Identity())
    
    def forward(self, x, radiomics_features=None, extra_params=None):
        if self.task in ['regression', 'classification']:
            x = self.pool(x)
            x = torch.flatten(x, 1)
            if radiomics_features is not None and self.radiomics_dim > 0:
                if radiomics_features.shape[1] != self.radiomics_dim:
                    raise ValueError(f"Radiomics features dimension ({radiomics_features.shape[1]}) does not match expected ({self.radiomics_dim})")
                x = torch.cat([x, radiomics_features], dim=1)
            if extra_params is not None and self.extra_params_dim > 0:
                if extra_params.shape[1] != self.extra_params_dim:
                    raise ValueError(f"Extra parameters dimension ({extra_params.shape[1]}) does not match expected ({self.extra_params_dim})")
                x = torch.cat([x, extra_params], dim=1)
            for layer in self.fc_layers:
                x = layer(x)
            return torch.sigmoid(x) if self.task == 'classification' else x
        else:  # segmentation
            if radiomics_features is not None and self.radiomics_dim > 0:
                radiomics_features = radiomics_features.view(x.shape[0], self.radiomics_dim, *[1]*self.dimension)
                radiomics_features = radiomics_features.repeat(1, 1, *x.shape[2:])
                x = torch.cat([x, radiomics_features], dim=1)
            if extra_params is not None and self.extra_params_dim > 0:
                extra_params = extra_params.view(x.shape[0], self.extra_params_dim, *[1]*self.dimension)
                extra_params = extra_params.repeat(1, 1, *x.shape[2:])
                x = torch.cat([x, extra_params], dim=1)
            logits = self.head(x)
            return logits  # return raw logits for CrossEntropyLoss

class MapToMapHead(nn.Module):
    """
    Head for image-to-image translation tasks (map-to-map).
    
    Performs dense pixel/voxel-level prediction for tasks like:
    - Image-to-image translation
    - Denoising / artifact removal
    - Image enhancement / super-resolution preparation
    - Multi-modal synthesis
    
    Args:
        in_channels: Number of input channels from backbone
        out_channels: Number of output channels (typically equals input for translation)
        dimension: Spatial dimension (1, 2, or 3)
        dropout_rate: Dropout probability
        activation: Final activation ('sigmoid', 'tanh', 'none')
        bias: Whether to use bias in conv layers
        extra_params_dim: Dimension of extra parameters (0 to disable)
    """
    def __init__(self, in_channels, out_channels, dimension=2, dropout_rate=0.0,
                 activation='none', bias=False, extra_params_dim=0):
        super().__init__()
        
        ConvNd, _, _, _, _, _ = getNdTools(dimension)
        self.dimension = dimension
        self.extra_params_dim = extra_params_dim
        self.activation_name = activation.lower()
        
        # 1×1 convolution to map features to output channels
        self.head = ConvNd(in_channels + extra_params_dim, out_channels, kernel_size=1, bias=bias)
        
        # Optional activation
        activation_dict = {
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
            'relu': nn.ReLU(),
            'none': nn.Identity()
        }
        self.activation = activation_dict.get(activation.lower(), nn.Identity())
    
    def forward(self, x, extra_params=None):
        """
        Args:
            x: Feature maps [B, C, ...]
            extra_params: Extra parameters [B, extra_params_dim] (optional)
        
        Returns:
            Reconstructed image [B, out_channels, ...]
        """
        if extra_params is not None and self.extra_params_dim > 0:
            if extra_params.shape[1] != self.extra_params_dim:
                raise ValueError(f"Extra parameters dimension ({extra_params.shape[1]}) does not match expected ({self.extra_params_dim})")
            # Expand extra_params to spatial dimensions
            extra_params = extra_params.view(x.shape[0], self.extra_params_dim, *[1]*self.dimension)
            extra_params = extra_params.repeat(1, 1, *x.shape[2:])
            x = torch.cat([x, extra_params], dim=1)
        
        logits = self.head(x)
        return self.activation(logits)

class EMUNetMapToMap(nn.Module):
    """
    Enhanced Multi-task U-Net for image-to-image translation (map-to-map).
    
    Combines symmetric encoder-decoder architecture with skip connections
    for high-quality image reconstruction and translation tasks.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels (typically == in_channels for translation)
        dimension: Spatial dimension (1, 2, or 3)
        num_filters: List of filter counts per level
        activation_final: Final activation ('sigmoid', 'tanh', 'none')
        use_batchnorm: Whether to use batch normalization
        activation: Hidden layer activation
        dropout_rate: Dropout probability
        leaky_slope: Negative slope for LeakyReLU
        bias: Whether to use bias
        use_attention: Whether to use CBAM attention
        use_skip_attention: Whether to apply attention to skip connections
        extra_params_dim: Dimension of extra parameters
    
    Example:
        >>> model = EMUNetMapToMap(
        ...     in_channels=1, out_channels=1, dimension=2,
        ...     num_filters=[64, 128, 256, 512],
        ...     activation_final='sigmoid'
        ... )
        >>> x = torch.randn(2, 1, 256, 256)
        >>> output = model(x)  # [2, 1, 256, 256]
    """
    def __init__(self, in_channels, out_channels, dimension=2, num_filters=[64, 128, 256, 512],
                 activation_final='none', use_batchnorm=True,
                 activation='leaky_relu', dropout_rate=0.0, leaky_slope=0.1, bias=False,
                 use_attention=True, use_skip_attention=False, extra_params_dim=0,
                 use_radiomics=False, num_bins=256, radii=[1]):
        super().__init__()
        
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("Channels must be positive")
        
        self.dimension = dimension
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.task = 'map-to-map'
        self.use_skip_attention = use_skip_attention
        self.extra_params_dim = extra_params_dim
        self.use_radiomics = use_radiomics
        self.num_bins = num_bins
        self.radii = radii
        
        # Use UNetBase as encoder-decoder backbone
        self.base = UNetBase(in_channels, num_filters, dimension, 3, use_batchnorm,
                            activation, dropout_rate, leaky_slope, bias, False, use_attention)
        
        # Radiomics: 24 FOS + (3 features × directions × radii)
        num_directions = dimension
        radiomics_dim = (24 + 3 * num_directions * len(radii)) * in_channels if use_radiomics else 0
        
        # Map-to-map head with radiomics support
        self.head = MapToMapHead(num_filters[0] + radiomics_dim, out_channels, dimension, dropout_rate,
                                activation_final, bias, extra_params_dim)
        
        # Optional fusion: gate feature maps with extra_params for conditioned generation
        self.use_fusion = extra_params_dim > 0
        if self.use_fusion:
            self.fusion = FusionHead(num_filters[0], extra_params_dim)
    
    def forward(self, x, extra_params=None):
        """
        Args:
            x: Input image [B, in_channels, ...]
            extra_params: Extra parameters [B, extra_params_dim] (optional)
        
        Returns:
            Reconstructed image [B, out_channels, ...]
        """
        expected_dims = self.dimension + 2
        if x.dim() != expected_dims:
            raise ValueError(f"Expected {self.dimension}D input with shape [B, C, ...], got {x.shape}")
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, got {x.shape[1]}")
        
        if extra_params is not None and self.extra_params_dim > 0:
            if extra_params.shape[1] != self.extra_params_dim:
                raise ValueError(f"Provided extra_params dim ({extra_params.shape[1]}) does not match initialized extra_params_dim ({self.extra_params_dim})")
        
        # Compute radiomics if enabled
        radiomics_features = None
        if self.use_radiomics:
            radiomics_features = self._compute_radiomics(x)
        
        # Process through base network
        features = self.base(x)
        
        # Concatenate radiomics features if available
        if radiomics_features is not None:
            # Broadcast radiomics to spatial dimensions
            radiomics_features = radiomics_features.view(features.shape[0], -1, *[1]*self.dimension)
            radiomics_features = radiomics_features.repeat(1, 1, *features.shape[2:])
            features = torch.cat([features, radiomics_features], dim=1)
        
        return self.head(features, extra_params)
    
    def _compute_radiomics(self, x):
        """
        Compute radiomics features (FOS + GLCM) for each channel.
        Features are standardized per sample for stability.
        """
        stats_features = []
        for i in range(x.shape[0]):
            channelfeatures = []
            for j in range(self.in_channels):
                fos = calculate_fos_features(x[i, j], num_bins=self.num_bins)
                glcm = calculate_simple_glcm_features(x[i, j], radii=self.radii, dimension=self.dimension)
                combined = torch.cat((fos, glcm))
                # Standardize features for better numerical stability
                combined = (combined - combined.mean()) / (combined.std() + 1e-6)
                channelfeatures.append(combined)
            stats_features.append(torch.cat(channelfeatures))
        return torch.stack(stats_features)
    
    def extract_features(self, x):
        """Extract features from bottleneck for visualization/analysis."""
        bottleneck_features, skip_connections = self.base.forward_features(x)
        radiomics_features = self._compute_radiomics(x) if self.use_radiomics else None
        return bottleneck_features, radiomics_features

class EMUNetPPMapToMap(nn.Module):
    """
    Enhanced UNet++ for image-to-image translation with dense skip connections.
    
    The UNet++ architecture provides multiple decoding paths, improving
    the quality of reconstructed images compared to standard UNet.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        dimension: Spatial dimension (1, 2, or 3)
        num_filters: List of filter counts per level
        activation_final: Final activation ('sigmoid', 'tanh', 'none')
        use_batchnorm: Whether to use batch normalization
        activation: Hidden layer activation
        dropout_rate: Dropout probability
        leaky_slope: Negative slope for LeakyReLU
        bias: Whether to use bias
        use_attention: Whether to use CBAM attention
        extra_params_dim: Dimension of extra parameters
    
    Example:
        >>> model = EMUNetPPMapToMap(
        ...     in_channels=1, out_channels=1, dimension=3,
        ...     num_filters=[32, 64, 128]
        ... )
        >>> x = torch.randn(1, 1, 64, 64, 64)
        >>> output = model(x)  # [1, 1, 64, 64, 64]
    """
    def __init__(self, in_channels, out_channels, dimension=2, num_filters=[64, 128, 256],
                 activation_final='none', use_batchnorm=True, activation='leaky_relu',
                 dropout_rate=0.0, leaky_slope=0.1, bias=False, use_attention=True,
                 extra_params_dim=0, use_radiomics=False, num_bins=256, radii=[1]):
        super().__init__()
        
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("Channels must be positive")
        
        self.dimension = dimension
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.task = 'map-to-map'
        self.extra_params_dim = extra_params_dim
        self.use_radiomics = use_radiomics
        self.num_bins = num_bins
        self.radii = radii
        
        # Use UNetPPBase as encoder-decoder backbone
        self.base = UNetPPBase(in_channels, num_filters, dimension, 3, use_batchnorm,
                              activation, dropout_rate, leaky_slope, bias, use_attention)
        
        # Radiomics: 24 FOS + (3 features × directions × radii)
        num_directions = dimension
        radiomics_dim = (24 + 3 * num_directions * len(radii)) * in_channels if use_radiomics else 0
        
        # Map-to-map head with radiomics support
        self.head = MapToMapHead(num_filters[0] + radiomics_dim, out_channels, dimension, dropout_rate,
                                activation_final, bias, extra_params_dim)
        
        # Optional fusion: gate feature maps with extra_params for conditioned generation
        self.use_fusion = extra_params_dim > 0
        if self.use_fusion:
            self.fusion = FusionHead(num_filters[0], extra_params_dim)
    
    def _compute_radiomics(self, x):
        """
        Compute radiomics features (FOS + GLCM) for each channel.
        Features are standardized per sample for stability.
        """
        stats_features = []
        for i in range(x.shape[0]):
            channelfeatures = []
            for j in range(self.in_channels):
                fos = calculate_fos_features(x[i, j], num_bins=self.num_bins)
                glcm = calculate_simple_glcm_features(x[i, j], radii=self.radii, dimension=self.dimension)
                combined = torch.cat((fos, glcm))
                # Standardize features for better numerical stability
                combined = (combined - combined.mean()) / (combined.std() + 1e-6)
                channelfeatures.append(combined)
            stats_features.append(torch.cat(channelfeatures))
        return torch.stack(stats_features)
    
    def forward(self, x, extra_params=None):
        """
        Args:
            x: Input image [B, in_channels, ...]
            extra_params: Extra parameters [B, extra_params_dim] (optional)
        
        Returns:
            Reconstructed image [B, out_channels, ...]
        """
        expected_dims = self.dimension + 2
        if x.dim() != expected_dims:
            raise ValueError(f"Expected {self.dimension}D input with shape [B, C, ...], got {x.shape}")
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, got {x.shape[1]}")
        
        if extra_params is not None and self.extra_params_dim > 0:
            if extra_params.shape[1] != self.extra_params_dim:
                raise ValueError(f"Provided extra_params dim ({extra_params.shape[1]}) does not match initialized extra_params_dim ({self.extra_params_dim})")
        
        # Compute radiomics if enabled
        radiomics_features = None
        if self.use_radiomics:
            radiomics_features = self._compute_radiomics(x)
        
        # Process through base network
        features = self.base(x)
        
        # Apply fusion gating if extra_params provided (for conditioned generation)
        if extra_params is not None and self.use_fusion:
            features = self.fusion(features, extra_params)
        
        # Concatenate radiomics features if available
        if radiomics_features is not None:
            # Broadcast radiomics to spatial dimensions
            radiomics_features = radiomics_features.view(features.shape[0], -1, *[1]*self.dimension)
            radiomics_features = radiomics_features.repeat(1, 1, *features.shape[2:])
            features = torch.cat([features, radiomics_features], dim=1)
        
        return self.head(features, extra_params)
    
    def extract_features(self, x):
        """Extract features from bottleneck for visualization/analysis."""
        bottleneck_features = self.base.forward_features(x)
        radiomics_features = self._compute_radiomics(x) if self.use_radiomics else None
        return bottleneck_features, radiomics_features

class SkipConnectionAligner(nn.Module):
    """
    Utility to handle spatial misalignment in skip connections.
    
    Provides flexible padding/cropping strategies for cases where
    encoder and decoder feature maps have different spatial dimensions.
    
    Strategies:
        'pad': Zero-pad decoder features to match encoder
        'crop': Crop encoder features to match decoder
        'interpolate': Interpolate decoder features to match encoder
    """
    def __init__(self, strategy='interpolate'):
        super().__init__()
        if strategy not in ['pad', 'crop', 'interpolate']:
            raise ValueError(f"Unknown strategy: {strategy}. Must be one of 'pad', 'crop', 'interpolate'")
        self.strategy = strategy
    
    def forward(self, encoder_feat, decoder_feat, dimension):
        """
        Align decoder_feat to match encoder_feat spatial dimensions.
        
        Args:
            encoder_feat: Feature map from encoder [B, C, ...]
            decoder_feat: Feature map from decoder [B, C, ...]
            dimension: Number of spatial dimensions (1, 2, or 3)
        
        Returns:
            Aligned decoder feature [B, C, ...] matching encoder spatial shape
        """
        target_shape = encoder_feat.shape[2:]
        current_shape = decoder_feat.shape[2:]
        
        if current_shape == target_shape:
            return decoder_feat
        
        if self.strategy == 'interpolate':
            mode = 'linear' if dimension == 1 else 'bilinear' if dimension == 2 else 'trilinear'
            return F.interpolate(decoder_feat, size=target_shape, mode=mode, align_corners=False)
        
        elif self.strategy == 'pad':
            # Calculate padding needed for each dimension
            padding = []
            for curr, targ in zip(reversed(current_shape), reversed(target_shape)):
                pad_total = targ - curr
                pad_before = pad_total // 2
                pad_after = pad_total - pad_before
                padding.extend([pad_before, pad_after])
            return F.pad(decoder_feat, padding, mode='constant', value=0)
        
        else:  # 'crop'
            slices = [slice(None), slice(None)]  # batch and channel
            for curr, targ in zip(current_shape, target_shape):
                start = (curr - targ) // 2
                slices.append(slice(start, start + targ))
            return decoder_feat[tuple(slices)]


class EMUNet(nn.Module):
    """
    Enhanced U-Net for semantic segmentation with optional radiomics and extra parameters.
    
    This model is specifically designed for pixel/voxel-level segmentation tasks.
    For classification or regression, use EMLeNet or EMResNet instead.
    
    Compatible with pyable-dataloader TrenoDataset output format.
    
    Example:
        >>> from treno.loaders import TrenoDataset
        >>> from treno.models import EMUNet
        >>> 
        >>> # Create dataset
        >>> dataset = TrenoDataset(manifest='data.json', target_size=[64, 64, 64])
        >>> 
        >>> # Create model for segmentation
        >>> model = EMUNet(
        ...     in_channels=1,
        ...     out_channels=4,
        ...     dimension=3,
        ...     use_radiomics=True,
        ...     extra_params_dim=3  # e.g., age, TR, TE
        ... )
        >>> 
        >>> # Forward pass
        >>> batch = dataset[0]
        >>> output = model(batch['images'], extra_params=batch.get('aux_data'))
    """
    def __init__(self, in_channels, out_channels, dimension=2, num_filters=[64, 128, 256],
                 use_batchnorm=True, activation='leaky_relu',
                 dropout_rate=0.0, leaky_slope=0.1, bias=False,
                 extra_params_dim=0, use_residual=False, use_attention=True,
                 use_radiomics=False, num_bins=256, radii=[1], reduction=2,
                 use_skip_attention=False):
        super().__init__()
        
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("Channels must be positive")
        if extra_params_dim < 0:
            raise ValueError("extra_params_dim must be non-negative")
            
        self.dimension = dimension
        self.use_radiomics = use_radiomics
        self.num_bins = num_bins
        self.radii = radii
        self.extra_params_dim = extra_params_dim
        self.in_channels = in_channels
        self.task = 'segmentation'  # Fixed task
        self.base = UNetBase(
            in_channels, num_filters, dimension, 3, use_batchnorm,
            activation, dropout_rate, leaky_slope, bias, use_residual, use_attention, reduction,
            use_skip_attention
        )
        
        # Radiomics: 24 FOS + (3 features × directions × radii)
        # directions: 1D=1, 2D=2, 3D=3
        num_directions = dimension
        radiomics_dim = (24 + 3 * num_directions * len(radii)) * in_channels if use_radiomics else 0
        
        self.head = NetworkHead(
            num_filters[0], out_channels, dimension, 'segmentation', [],
            dropout_rate, activation, leaky_slope, bias, radiomics_dim, extra_params_dim
        )
        # Optional fusion: gate UNet output feature maps with extra_params
        self.use_fusion = extra_params_dim > 0
        if self.use_fusion:
            self.fusion = FusionHead(num_filters[0], extra_params_dim)
        
    def forward(self, x, extra_params=None):
        # Validate input dimensions
        expected_dims = self.dimension + 2  # +2 for batch and channel
        if x.dim() != expected_dims:
            raise ValueError(
                f"Expected {self.dimension}D input with shape [B, C, "
                f"{'D, ' if self.dimension == 3 else ''}H, W], got shape {x.shape}"
            )
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, got {x.shape[1]}"
            )
        
        radiomics_features = None
        if self.use_radiomics:
            radiomics_features = self._compute_radiomics(x)
        x = self.base(x)
        # validate extra_params dimensions before applying fusion
        if extra_params is not None and self.extra_params_dim > 0:
            if extra_params.shape[1] != self.extra_params_dim:
                raise ValueError(f"Provided extra_params dim ({extra_params.shape[1]}) does not match initialized extra_params_dim ({self.extra_params_dim})")
        # apply fusion gating to UNet output feature maps if requested
        if extra_params is not None and self.use_fusion:
            x = self.fusion(x, extra_params)
        return self.head(x, radiomics_features, extra_params)
    
    def _compute_radiomics(self, x):
        """
        Compute radiomics features (FOS + GLCM) for each channel.
        Features are standardized per sample for stability.
        """
        stats_features = []
        for i in range(x.shape[0]):
            channelfeatures = []
            for j in range(self.in_channels):
                fos = calculate_fos_features(x[i, j], num_bins=self.num_bins)
                glcm = calculate_simple_glcm_features(x[i, j], radii=self.radii, dimension=self.dimension)
                combined = torch.cat((fos, glcm))
                # Standardize features for better numerical stability
                combined = (combined - combined.mean()) / (combined.std() + 1e-6)
                channelfeatures.append(combined)
            stats_features.append(torch.cat(channelfeatures))
        return torch.stack(stats_features)
    
    def extract_features(self, x):
        bottleneck_features, skip_connections = self.base.forward_features(x)
        bottleneck_features = self.base.bottleneck(bottleneck_features)
        radiomics_features = self._compute_radiomics(x) if self.use_radiomics else None
        return bottleneck_features, skip_connections, radiomics_features

class EMLeNet(nn.Module):
    """
    Enhanced Multi-task LeNet architecture with optional radiomics and extra parameters.
    
    Compatible with pyable-dataloader TrenoDataset output format.
    
    Example:
        >>> from treno.loaders import TrenoDataset
        >>> from treno.models import EMLeNet
        >>> 
        >>> # Create dataset
        >>> dataset = TrenoDataset(manifest='data.json', target_size=[32, 32])
        >>> 
        >>> # Create model for classification
        >>> model = EMLeNet(
        ...     in_channels=1,
        ...     out_channels=10,
        ...     dimension=2,
        ...     task='classification',
        ...     use_radiomics=True
        ... )
        >>> 
        >>> # Forward pass
        >>> batch = dataset[0]
        >>> output = model(batch['images'])
    """
    def __init__(self, in_channels, out_channels, dimension=2, num_filters=[16, 32, 64],
                 task='regression', use_batchnorm=True, activation='leaky_relu',
                 dropout_rate=0.0, leaky_slope=0.1, bias=False, fc_layers=[1024, 512],
                 extra_params_dim=0, use_residual=False, use_attention=True,
                 use_radiomics=False, num_bins=256, radii=[1],reduction=2):
        super().__init__()
        
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("Channels must be positive")
        if extra_params_dim < 0:
            raise ValueError("extra_params_dim must be non-negative")
            
        self.dimension = dimension
        self.use_radiomics = use_radiomics
        self.num_bins = num_bins
        self.radii = radii
        self.extra_params_dim = extra_params_dim
        self.in_channels = in_channels
        self.base = LeNetBase(
            in_channels, num_filters, dimension, 3, use_batchnorm,
            activation, dropout_rate, leaky_slope, bias, use_residual, use_attention,reduction
        )
        
        # Radiomics: 24 FOS + (3 features × directions × radii)
        num_directions = dimension
        radiomics_dim = (24 + 3 * num_directions * len(radii)) * in_channels if use_radiomics else 0
        
        self.head = NetworkHead(
            num_filters[-2], out_channels, dimension, task, fc_layers,
            dropout_rate, activation, leaky_slope, bias, radiomics_dim, extra_params_dim
        )
        # Optional fusion: gate LeNet output feature maps with extra_params
        self.use_fusion = extra_params_dim > 0
        if self.use_fusion:
            self.fusion = FusionHead(num_filters[-2], extra_params_dim)
        
    def forward(self, x, extra_params=None):
        radiomics_features = None
        if self.use_radiomics:
            radiomics_features = self._compute_radiomics(x)
        x = self.base(x)
        # validate extra_params dimensions before applying fusion
        if extra_params is not None and self.extra_params_dim > 0:
            if extra_params.shape[1] != self.extra_params_dim:
                raise ValueError(f"Provided extra_params dim ({extra_params.shape[1]}) does not match initialized extra_params_dim ({self.extra_params_dim})")
        # apply fusion gating to LeNet feature maps if requested
        if extra_params is not None and self.use_fusion:
            x = self.fusion(x, extra_params)
        return self.head(x, radiomics_features, extra_params)
    
    def _compute_radiomics(self, x):
        stats_features = []
        for i in range(x.shape[0]):
            channelfeatures = []
            for j in range(self.in_channels):
                fos = calculate_fos_features(x[i, j], num_bins=self.num_bins)
                glcm = calculate_simple_glcm_features(x[i, j], radii=self.radii, dimension=self.dimension)
                combined = torch.cat((fos, glcm))
                combined = combined / (torch.max(torch.abs(combined)) + 1e-6)
                channelfeatures.append(combined)
            stats_features.append(torch.cat(channelfeatures))
        return torch.stack(stats_features)
    
    def extract_features(self, x):
        conv_features = self.base.forward_features(x)
        radiomics_features = self._compute_radiomics(x) if self.use_radiomics else None
        return conv_features, radiomics_features

# ============================================================================
# TRAINING UTILITIES
# ============================================================================

class EarlyStopping:
    """
    Early stopping to stop training when validation loss doesn't improve.
    
    Args:
        patience: How many epochs to wait after last improvement
        verbose: Whether to print messages
        delta: Minimum change to qualify as improvement
        path: Path to save checkpoint
        
    Example:
        >>> early_stopping = EarlyStopping(patience=7, verbose=True)
        >>> 
        >>> for epoch in range(num_epochs):
        ...     train_loss = train_one_epoch(model, train_loader)
        ...     val_loss = validate(model, val_loader)
        ...     
        ...     early_stopping(val_loss, model)
        ...     if early_stopping.early_stop:
        ...         print("Early stopping triggered")
        ...         break
    """
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')

    def __call__(self, val_loss, model):
        score = -val_loss  # Higher score = better

        if self.best_score is None:
            # First validation step
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score > self.best_score + self.delta:
            # Significant improvement
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
        else:
            # No improvement
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, val_loss, model):
        """Save model when validation loss improves."""
        if self.verbose:
            print(f'Validation loss improved ({self.val_loss_min:.4f} → {val_loss:.4f}). Saving model...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


class ModelCheckpoint:
    """
    Save model checkpoints with epoch and training information.
    
    Args:
        save_dir: Directory to save checkpoints
        monitor: Metric to monitor ('val_loss' or 'val_acc')
        mode: 'min' or 'max' (whether lower or higher is better)
        save_best_only: Only save when metric improves
        
    Example:
        >>> checkpoint = ModelCheckpoint(save_dir='./checkpoints', monitor='val_loss')
        >>> 
        >>> for epoch in range(num_epochs):
        ...     train_loss = train_one_epoch(model, train_loader)
        ...     val_loss = validate(model, val_loader)
        ...     
        ...     checkpoint.save(model, optimizer, epoch, val_loss)
    """
    def __init__(self, save_dir='./checkpoints', monitor='val_loss', mode='min', save_best_only=True):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.best_metric = float('inf') if mode == 'min' else float('-inf')
        
    def save(self, model, optimizer, epoch, metrics, filename=None):
        """
        Save model checkpoint.
        
        Args:
            model: PyTorch model
            optimizer: PyTorch optimizer
            epoch: Current epoch number
            metrics: Dict of metrics (e.g., {'val_loss': 0.5, 'val_acc': 0.9})
            filename: Optional custom filename
        """
        metric_value = metrics.get(self.monitor, None)
        
        if metric_value is None:
            print(f"Warning: Metric '{self.monitor}' not found in metrics dict")
            return
        
        is_best = False
        if self.mode == 'min':
            is_best = metric_value < self.best_metric
        else:
            is_best = metric_value > self.best_metric
            
        if is_best:
            self.best_metric = metric_value
            
        if not self.save_best_only or is_best:
            if filename is None:
                filename = f'checkpoint_epoch_{epoch:03d}.pt'
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metrics,
                'best_metric': self.best_metric
            }
            
            save_path = self.save_dir / filename
            torch.save(checkpoint, save_path)
            
            if is_best:
                # Also save as best model
                best_path = self.save_dir / 'best_model.pt'
                torch.save(checkpoint, best_path)
                print(f'✓ New best {self.monitor}: {metric_value:.4f} (saved to {best_path})')


# ============================================================================
# Model I/O - Now imported from pyable-ml
# ============================================================================

from pyml.io import save_model, load_model, save_checkpoint, load_checkpoint

# Note: The pyable-ml versions have slightly different signatures:
# - save_model(model, path) works the same
# - load_model(path, model_class=instance, device='cpu') requires model_class parameter
# - save_checkpoint and load_checkpoint work similarly but return dict instead of tuple

__all__ = ['save_model', 'load_model', 'save_checkpoint', 'load_checkpoint']


class TrainingHistory:
    """
    Track training metrics over epochs.
    
    Example:
        >>> history = TrainingHistory()
        >>> 
        >>> for epoch in range(num_epochs):
        ...     train_loss = train_one_epoch(model, train_loader)
        ...     val_loss = validate(model, val_loader)
        ...     
        ...     history.add_epoch({
        ...         'train_loss': train_loss,
        ...         'val_loss': val_loss,
        ...         'learning_rate': optimizer.param_groups[0]['lr']
        ...     })
        ...     
        ...     history.plot(save_path='training_curves.png')
    """
    def __init__(self):
        self.history = {}
        
    def add_epoch(self, metrics):
        """Add metrics for one epoch."""
        for key, value in metrics.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(value)
    
    def get_metric(self, metric_name):
        """Get all values for a specific metric."""
        return self.history.get(metric_name, [])
    
    def plot(self, metrics=None, save_path=None):
        """
        Plot training curves.
        
        Args:
            metrics: List of metric names to plot (None = plot all)
            save_path: Path to save figure
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Warning: matplotlib not installed, cannot plot")
            return
        
        if metrics is None:
            metrics = list(self.history.keys())
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for metric in metrics:
            if metric in self.history:
                epochs = range(1, len(self.history[metric]) + 1)
                ax.plot(epochs, self.history[metric], label=metric, marker='o')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Value')
        ax.set_title('Training History')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def save(self, path):
        """Save history to file."""
        import json
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"✓ History saved to {path}")
    
    def load(self, path):
        """Load history from file."""
        import json
        with open(path, 'r') as f:
            self.history = json.load(f)
        print(f"✓ History loaded from {path}")


# ============================================================================
# SIMPLER ATTENTION MECHANISM (alternative to CBAM)
# ============================================================================

class SimpleAttention(nn.Module):
    """
    Simpler channel attention mechanism (alternative to CBAM).
    Uses global average pooling + small MLP.
    
    This is lighter weight than CBAM but still effective.
    """
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels // reduction, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        attn = self.attention(x)
        return x * attn




# ============================================================================
# ATTENTION GATE (Attention U-Net style)
# ============================================================================

class AttentionGate(nn.Module):
    """Attention gate for skip connections in U-Net (dimension-agnostic)."""
    def __init__(self, in_channels_x, in_channels_g, inter_channels, dimension=2):
        super().__init__()
        ConvNd, _, _, BatchNormNd, _, _ = getNdTools(dimension)
        self.theta_x = ConvNd(in_channels_x, inter_channels, kernel_size=2, stride=2, bias=False)
        self.phi_g = ConvNd(in_channels_g, inter_channels, kernel_size=1, bias=False)
        self.psi = ConvNd(inter_channels, 1, kernel_size=1, bias=True)
        self.bn = BatchNormNd(inter_channels)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x, g):
        # x: skip connection, g: gating (decoder feature)
        x1 = self.theta_x(x)
        g1 = self.phi_g(g)
        # align shapes
        if x1.shape[2:] != g1.shape[2:]:
            mode = 'linear' if len(x1.shape) == 3 else 'bilinear' if len(x1.shape) == 4 else 'trilinear'
            g1 = F.interpolate(g1, size=x1.shape[2:], mode=mode, align_corners=False)
        z = self.relu(self.bn(x1 + g1))
        attn = self.sigmoid(self.psi(z))
        # upsample attn back to x size
        if attn.shape[2:] != x.shape[2:]:
            mode = 'linear' if len(attn.shape) == 3 else 'bilinear' if len(attn.shape) == 4 else 'trilinear'
            attn = F.interpolate(attn, size=x.shape[2:], mode=mode, align_corners=False)
        return x * attn


# ============================================================================
# FUSION HEAD (clinical/imaging gated fusion)
# ============================================================================

class FusionHead(nn.Module):
    """Fusion module that uses extra scalar params to gate image feature maps.

    It computes a small MLP from extra_params -> per-channel gates, applies sigmoid
    and multiplies (1 + gate) * feature_map so image features are modulated by
    clinical signals. Dimension-agnostic: gates are broadcast over spatial dims.
    """
    def __init__(self, in_channels, extra_dim, hidden=[64, 64], activation='relu'):
        super().__init__()
        layers = []
        input_dim = extra_dim
        for h in hidden:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU(inplace=True) if activation == 'relu' else nn.LeakyReLU(0.1, inplace=True))
            input_dim = h
        layers.append(nn.Linear(input_dim, in_channels))
        self.mlp = nn.Sequential(*layers)
        self.sigmoid = nn.Sigmoid()

    def forward(self, feat, extra_params):
        """Apply gating: feat shape [B, C, ...], extra_params [B, D] -> returns gated feat"""
        if extra_params is None:
            return feat
        gates = self.mlp(extra_params)  # [B, C]
        gates = self.sigmoid(gates).unsqueeze(-1)
        # expand to spatial dims
        spatial_dims = feat.dim() - 2
        for _ in range(spatial_dims - 1):
            gates = gates.unsqueeze(-1)
        # gates shape [B, C, 1, ...]
        return feat * (1.0 + gates)


# ============================================================================
# RESNET ENCODERS (1D/2D/3D)
# ============================================================================

class BasicBlockNd(nn.Module):
    """Dimension-agnostic Basic Residual Block (like ResNet-18/34)."""
    expansion = 1
    def __init__(self, in_channels, out_channels, dimension=2, stride=1,
                 use_batchnorm=True, activation='relu', bias=False):
        super().__init__()
        ConvNd, _, _, BatchNormNd, _, _ = getNdTools(dimension)
        self.use_bn = use_batchnorm
        padding = 1
        self.conv1 = ConvNd(in_channels, out_channels, kernel_size=3, stride=stride,
                            padding=padding, bias=bias)
        self.bn1 = BatchNormNd(out_channels) if use_batchnorm else nn.Identity()
        self.act = nn.ReLU(inplace=True) if activation == 'relu' else nn.LeakyReLU(0.1, inplace=True)
        self.conv2 = ConvNd(out_channels, out_channels, kernel_size=3, stride=1,
                            padding=padding, bias=bias)
        self.bn2 = BatchNormNd(out_channels) if use_batchnorm else nn.Identity()
        
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                ConvNd(in_channels, out_channels, kernel_size=1, stride=stride, bias=bias),
                BatchNormNd(out_channels) if use_batchnorm else nn.Identity()
            )
    
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.act(out)
        return out

class ResNetEncoder(nn.Module):
    """ResNet-style encoder with AdaptiveAvgPool output for heads.
    Supports 1D, 2D, 3D by passing dimension.
    """
    def __init__(self, in_channels, dimension=2, layers=(2, 2, 2, 2), base_width=64,
                 use_batchnorm=True, activation='relu', bias=False):
        super().__init__()
        ConvNd, _, MaxPoolNd, BatchNormNd, _, _ = getNdTools(dimension)
        self.dimension = dimension
        self.inplanes = base_width
        self.use_bn = use_batchnorm
        
        # Stem
        self.conv1 = ConvNd(in_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=bias)
        self.bn1 = BatchNormNd(self.inplanes) if use_batchnorm else nn.Identity()
        self.act = nn.ReLU(inplace=True) if activation == 'relu' else nn.LeakyReLU(0.1, inplace=True)
        self.pool = MaxPoolNd(kernel_size=3, stride=2, padding=1)
        
        # Layers
        self.layer1 = self._make_layer(self.inplanes, base_width, layers[0], stride=1, activation=activation)
        self.layer2 = self._make_layer(base_width, base_width*2, layers[1], stride=2, activation=activation)
        self.layer3 = self._make_layer(base_width*2, base_width*4, layers[2], stride=2, activation=activation)
        self.layer4 = self._make_layer(base_width*4, base_width*8, layers[3], stride=2, activation=activation)
        
        # Output pool
        self.gap = {1: nn.AdaptiveAvgPool1d(1), 2: nn.AdaptiveAvgPool2d(1), 3: nn.AdaptiveAvgPool3d(1)}[dimension]
        
    def _make_layer(self, in_c, out_c, blocks, stride, activation):
        layers = []
        layers.append(BasicBlockNd(in_c, out_c, dimension=self.dimension, stride=stride,
                                   use_batchnorm=self.use_bn, activation=activation))
        for _ in range(1, blocks):
            layers.append(BasicBlockNd(out_c, out_c, dimension=self.dimension, stride=1,
                                       use_batchnorm=self.use_bn, activation=activation))
        return nn.Sequential(*layers)
    
    def forward_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.pool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x
    
    def forward(self, x):
        x = self.forward_features(x)
        x = self.gap(x)
        return x

class EMResNet(nn.Module):
    """EMResNet: ResNet encoder + NetworkHead for classification/regression.
    Supports radiomics and extra_params like EMUNet.
    """
    def __init__(self, in_channels, out_channels, dimension=2, task='classification',
                 layers=(2,2,2,2), base_width=64, use_batchnorm=True, activation='relu',
                 dropout_rate=0.0, bias=False, fc_layers=[1024, 512],
                 extra_params_dim=0, use_radiomics=False, num_bins=256, radii=[1]):
        super().__init__()
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("Channels must be positive")
        self.dimension = dimension
        self.in_channels = in_channels
        self.task = task
        self.use_radiomics = use_radiomics
        self.num_bins = num_bins
        self.radii = radii
        self.extra_params_dim = extra_params_dim
        
        self.encoder = ResNetEncoder(in_channels, dimension, layers, base_width,
                                     use_batchnorm, activation, bias)
        # Radiomics: 24 FOS + (3 features × directions × radii)
        num_directions = dimension
        radiomics_dim = (24 + 3 * num_directions * len(radii)) * in_channels if use_radiomics else 0
        # Encoder output channels = base_width*8
        self.head = NetworkHead(base_width*8, out_channels, dimension, task, fc_layers,
                                dropout_rate, 'leaky_relu' if activation=='relu' else activation,
                                0.1, bias, radiomics_dim, extra_params_dim)
        # Optional fusion: gate encoder feature maps with extra_params
        self.use_fusion = extra_params_dim > 0
        if self.use_fusion:
            self.fusion = FusionHead(base_width*8, extra_params_dim)
    
    def _compute_radiomics(self, x):
        stats_features = []
        for i in range(x.shape[0]):
            channelfeatures = []
            for j in range(self.in_channels):
                fos = calculate_fos_features(x[i, j], num_bins=self.num_bins)
                glcm = calculate_simple_glcm_features(x[i, j], radii=self.radii, dimension=self.dimension)
                combined = torch.cat((fos, glcm))
                combined = (combined - combined.mean()) / (combined.std() + 1e-6)
                channelfeatures.append(combined)
            stats_features.append(torch.cat(channelfeatures))
        return torch.stack(stats_features)
    
    def forward(self, x, extra_params=None):
        # Validate input dims
        expected_dims = self.dimension + 2
        if x.dim() != expected_dims:
            raise ValueError(f"Expected {self.dimension}D input with shape [B, C, ...], got {x.shape}")
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, got {x.shape[1]}")
        radiomics_features = None
        if self.use_radiomics:
            radiomics_features = self._compute_radiomics(x)
        x = self.encoder(x)
        # validate extra_params dimensions before applying fusion
        if extra_params is not None and self.extra_params_dim > 0:
            if extra_params.shape[1] != self.extra_params_dim:
                raise ValueError(f"Provided extra_params dim ({extra_params.shape[1]}) does not match initialized extra_params_dim ({self.extra_params_dim})")
        # apply fusion gating to encoder feature maps if requested
        if extra_params is not None and self.use_fusion:
            x = self.fusion(x, extra_params)
        return self.head(x, radiomics_features, extra_params)

    def extract_features(self, x):
        """Extract encoder features before global pooling and FC, plus radiomics.

        Returns (features, radiomics_features) where features is [B, C, ...].
        """
        features = self.encoder.forward_features(x)
        radiomics_features = self._compute_radiomics(x) if self.use_radiomics else None
        return features, radiomics_features

def get_deep_radiomics_features(model, x, extra_params=None, pool='avg', include_engineered=True):
    """Return a [B, F] vector of deep radiomics features for any model.

    - Uses model.extract_features(x) when available.
    - Applies adaptive pooling (avg or max) over spatial dimensions to flatten deep features.
    - Optionally concatenates engineered radiomics if available/enabled.

    Args:
        model: Treno model instance
        x: input tensor [B, C, ...]
        extra_params: optional extra params [B, E] (unused here; present for future gating compatibility)
        pool: 'avg' or 'max'
        include_engineered: whether to append engineered radiomics features if present

    Returns:
        Tensor of shape [B, F]
    """
    # Get deep features and engineered radiomics
    deep = None
    engineered = None

    if hasattr(model, 'extract_features') and callable(getattr(model, 'extract_features')):
        feats = model.extract_features(x)
        # Handle different tuple arities across models
        if isinstance(feats, tuple):
            # EMUNet returns (bottleneck_features, skip_connections, radiomics_features)
            if len(feats) == 3:
                deep, _, engineered = feats
            elif len(feats) == 2:
                deep, engineered = feats
            else:
                deep = feats[0]
        else:
            deep = feats
    else:
        # Fallback: try common attribute names
        if hasattr(model, 'base') and hasattr(model.base, 'forward_features'):
            deep = model.base.forward_features(x)
        elif hasattr(model, 'encoder') and hasattr(model.encoder, 'forward_features'):
            deep = model.encoder.forward_features(x)
        else:
            raise AttributeError("Model does not expose extractable features. Implement extract_features().")

    # Adaptive pooling to [B, C, 1, ...] then flatten to [B, C]
    if deep.dim() > 2:
        spatial_dims = deep.dim() - 2
        if pool == 'avg':
            pool_layer = {1: nn.AdaptiveAvgPool1d(1), 2: nn.AdaptiveAvgPool2d(1), 3: nn.AdaptiveAvgPool3d(1)}.get(spatial_dims)
        else:
            pool_layer = {1: nn.AdaptiveMaxPool1d(1), 2: nn.AdaptiveMaxPool2d(1), 3: nn.AdaptiveMaxPool3d(1)}.get(spatial_dims)
        if pool_layer is None:
            raise ValueError(f"Unsupported spatial dims: {spatial_dims}")
        deep = pool_layer(deep)
        deep = torch.flatten(deep, 1)

    # Concatenate engineered radiomics if requested and available
    if include_engineered and engineered is not None:
        if engineered.dim() > 2:
            engineered = torch.flatten(engineered, 1)
        # Ensure shapes are [B, *]
        if engineered.shape[0] != deep.shape[0]:
            raise ValueError("Batch size mismatch between deep and engineered features")
        deep = torch.cat([deep, engineered], dim=1)

    return deep


# ============================================================================
# MAIN (TESTS)
# ============================================================================

if __name__ == "__main__":
    # Test configurations
    configs = [
        # 1D U-Net with radiomics and 3 extra params (age, TR, TE)
        {
            'model': EMUNet,
            'kwargs': {
                'in_channels': 2,
                'out_channels': 1,
                'dimension': 1,
                'num_filters': [32, 64],
                'extra_params_dim': 3,  # age, TR, TE
                'use_radiomics': True,
                'num_bins': 128,
                'radii': [1, 2]
            },
            'input': torch.randn(2, 2, 128),
            'extra': torch.tensor([[25.0, 100.0, 5.0], [30.0, 150.0, 10.0]])  # [age, TR, TE]
        },
        # 2D LeNet without radiomics, with 2 extra params (age, TR)
        {
            'model': EMLeNet,
            'kwargs': {
                'in_channels': 1,
                'out_channels': 10,
                'dimension': 2,
                'num_filters': [16, 32, 64],
                'task': 'classification',
                'fc_layers': [512, 256],
                'extra_params_dim': 2,  # age, TR
                'use_radiomics': False
            },
            'input': torch.randn(2, 1, 32, 32),
            'extra': torch.tensor([[40.0, 200.0], [45.0, 250.0]])  # [age, TR]
        },
               {
            'model': EMLeNet,
            'kwargs': {
                'in_channels': 1,
                'out_channels': 10,
                'dimension': 2,
                'num_filters': [16, 32, 64],
                'task': 'classification',
                'fc_layers': [512, 256],
                'extra_params_dim': 2,  # age, TR
                'use_radiomics': True
            },
            'input': torch.randn(2, 1, 32, 32),
            'extra': torch.tensor([[40.0, 200.0], [45.0, 250.0]])  # [age, TR]
        },
                          {
            'model': EMLeNet,
            'kwargs': {
                'in_channels': 1,
                'out_channels': 10,
                'dimension': 2,
                'num_filters': [16, 32, 64],
                'task': 'regression',
                'fc_layers': [512, 256],
                'extra_params_dim': 2,  # age, TR
                'use_radiomics': True
            },
            'input': torch.randn(2, 1, 32, 32),
            'extra': torch.tensor([[40.0, 200.0], [45.0, 250.0]])  # [age, TR]
        },
        # 3D U-Net with radiomics, no extra params
        {
            'model': EMUNet,
            'kwargs': {
                'in_channels': 3,
                'out_channels': 4,
                'dimension': 3,
                'num_filters': [32, 64],
                'use_residual': True,
                'use_radiomics': True,
                'num_bins': 64,
                'radii': [1, 3]
            },
            'input': torch.randn(2, 3, 18, 16, 16),
            'extra': None
        }
    ]
    
    for config in configs:
        model = config['model'](**config['kwargs'])
        x = config['input']
        extra = config['extra']
        out = model(x, extra) if extra is not None else model(x)
        task_str = getattr(model, 'task', 'n/a')
        if isinstance(model, EMUNet):
            features, skip, radiomics = model.extract_features(x)
            print(f"{config['kwargs']['dimension']}D {task_str} output shape: {out.shape}")
            print(f"{config['kwargs']['dimension']}D Extracted bottleneck features shape: {features.shape}")
            print(f"{config['kwargs']['dimension']}D Skip connections: {[s.shape for s in skip]}")
            print(f"{config['kwargs']['dimension']}D Radiomics features shape: {radiomics.shape if radiomics is not None else 'None'}")
        else:
            features, radiomics = model.extract_features(x)
            print(f"{config['kwargs']['dimension']}D {task_str} output shape: {out.shape}")
            print(f"{config['kwargs']['dimension']}D Extracted features shape: {features.shape}")
            print(f"{config['kwargs']['dimension']}D Radiomics features shape: {radiomics.shape if radiomics is not None else 'None'}")
        print()