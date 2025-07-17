import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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
    """Calculate simplified GLCM-like features for a tensor across multiple radii."""
    glcm_features = []
    for radius in radii:
        if dimension == 1:
            x_shift = torch.roll(x, shifts=radius, dims=0)
            x_shift[:radius] = 0
        elif dimension == 2:
            x_shift = torch.roll(x, shifts=radius, dims=1)
            x_shift[:, :radius] = 0
        else:  # 3D
            x_shift = torch.roll(x, shifts=radius, dims=2)
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
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
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
                 use_attention=True,reduction=2):
        
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
        for filters in reversed(num_filters):
            self.ups.append(nn.Sequential(
                ConvTransposeNd(filters*2, filters, kernel_size=2, stride=2),
                BaseConvBlock(filters*2, filters, dimension, kernel_size, 1,
                            use_batchnorm, activation, dropout_rate, leaky_slope, 
                            bias, use_residual, use_attention,reduction)
            ))
            
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
            x = torch.cat([skip, x], dim=1)
            x = up[1](x)
        return x

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
    """Configurable network head for different tasks with optional radiomics and extra parameters."""
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

class EMUNet(nn.Module):
    """Enhanced Multi-task U-Net architecture with optional radiomics and extra parameters."""
    def __init__(self, in_channels, out_channels, dimension=2, num_filters=[64, 128, 256],
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
        self.base = UNetBase(
            in_channels, num_filters, dimension, 3, use_batchnorm,
            activation, dropout_rate, leaky_slope, bias, use_residual, use_attention,reduction
        )
        
        radiomics_dim = (24 + 3 * len(radii)) * in_channels if use_radiomics else 0
        
        self.head = NetworkHead(
            num_filters[0], out_channels, dimension, task, fc_layers,
            dropout_rate, activation, leaky_slope, bias, radiomics_dim, extra_params_dim
        )
        
    def forward(self, x, extra_params=None):
        radiomics_features = None
        if self.use_radiomics:
            radiomics_features = self._compute_radiomics(x)
        x = self.base(x)
        if extra_params is not None and self.extra_params_dim > 0:
            if extra_params.shape[1] != self.extra_params_dim:
                raise ValueError(f"Provided extra_params dim ({extra_params.shape[1]}) does not match initialized extra_params_dim ({self.extra_params_dim})")
        return self.head(x, radiomics_features, extra_params)
    
    def _compute_radiomics(self, x):
        stats_features = []
        for i in range(x.shape[0]):
            channelfeatures = []
            for j in range(self.in_channels):
                fos = calculate_fos_features(x[i, j], num_bins=self.num_bins)
                glcm = calculate_simple_glcm_features(x[i, j], radii=self.radii, dimension=self.dimension)
                combined = torch.cat((fos, glcm),)
                combined = combined / (torch.max(torch.abs(combined)) + 1e-6)
                channelfeatures.append(combined)
            stats_features.append(torch.cat(channelfeatures))
        return torch.stack(stats_features)
    
    def extract_features(self, x):
        bottleneck_features, skip_connections = self.base.forward_features(x)
        bottleneck_features = self.base.bottleneck(bottleneck_features)
        radiomics_features = self._compute_radiomics(x) if self.use_radiomics else None
        return bottleneck_features, skip_connections, radiomics_features

class EMLeNet(nn.Module):
    """Enhanced Multi-task LeNet architecture with optional radiomics and extra parameters."""
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
        
        radiomics_dim = (24 + 3 * len(radii)) * in_channels if use_radiomics else 0
        
        self.head = NetworkHead(
            num_filters[-2], out_channels, dimension, task, fc_layers,
            dropout_rate, activation, leaky_slope, bias, radiomics_dim, extra_params_dim
        )
        
    def forward(self, x, extra_params=None):
        radiomics_features = None
        if self.use_radiomics:
            radiomics_features = self._compute_radiomics(x)
        x = self.base(x)
        if extra_params is not None and self.extra_params_dim > 0:
            if extra_params.shape[1] != self.extra_params_dim:
                raise ValueError(f"Provided extra_params dim ({extra_params.shape[1]}) does not match initialized extra_params_dim ({self.extra_params_dim})")
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
                'task': 'regression',
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
                'task': 'segmentation',
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
        if isinstance(model, EMUNet):
            features, skip, radiomics = model.extract_features(x)
            print(f"{config['kwargs']['dimension']}D {config['kwargs']['task']} output shape: {out.shape}")
            print(f"{config['kwargs']['dimension']}D Extracted bottleneck features shape: {features.shape}")
            print(f"{config['kwargs']['dimension']}D Skip connections: {[s.shape for s in skip]}")
            print(f"{config['kwargs']['dimension']}D Radiomics features shape: {radiomics.shape if radiomics is not None else 'None'}")
        else:
            features, radiomics = model.extract_features(x)
            print(f"{config['kwargs']['dimension']}D {config['kwargs']['task']} output shape: {out.shape}")
            print(f"{config['kwargs']['dimension']}D Extracted features shape: {features.shape}")
            print(f"{config['kwargs']['dimension']}D Radiomics features shape: {radiomics.shape if radiomics is not None else 'None'}")
        print()