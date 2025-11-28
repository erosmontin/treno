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

class EMUNet(nn.Module):
    """
    Enhanced Multi-task U-Net architecture with optional radiomics and extra parameters.
    
    Compatible with pyable-dataloader TrenoDataset output format.
    
    Example:
        >>> from treno.loaders import TrenoDataset
        >>> from treno.models import EMUNet
        >>> 
        >>> # Create dataset
        >>> dataset = TrenoDataset(manifest='data.json', target_size=[64, 64, 64])
        >>> 
        >>> # Create model
        >>> model = EMUNet(
        ...     in_channels=1,
        ...     out_channels=4,
        ...     dimension=3,
        ...     task='segmentation',
        ...     use_radiomics=True,
        ...     extra_params_dim=3  # e.g., age, TR, TE
        ... )
        >>> 
        >>> # Forward pass
        >>> batch = dataset[0]
        >>> output = model(batch['images'], extra_params=batch.get('aux_data'))
    """
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
        self.task = task
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
        if extra_params is not None and self.extra_params_dim > 0:
            if extra_params.shape[1] != self.extra_params_dim:
                raise ValueError(f"Provided extra_params dim ({extra_params.shape[1]}) does not match initialized extra_params_dim ({self.extra_params_dim})")
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


def save_model(model, path):
    """Save model state dict."""
    torch.save(model.state_dict(), path)
    print(f"✓ Model saved to {path}")


def load_model(model, path, device='cpu'):
    """
    Load model state dict.
    
    Args:
        model: Model instance to load weights into
        path: Path to saved state dict
        device: Device to load model on
        
    Returns:
        Model with loaded weights
    """
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    print(f"✓ Model loaded from {path}")
    return model


def save_checkpoint(model, optimizer, epoch, loss, path):
    """
    Save complete training checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        epoch: Current epoch
        loss: Current loss value
        path: Save path
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, path)
    print(f"✓ Checkpoint saved to {path}")


def load_checkpoint(model, optimizer, path, device='cpu'):
    """
    Load complete training checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        path: Path to checkpoint
        device: Device to load on
        
    Returns:
        Tuple of (model, optimizer, epoch, loss)
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"✓ Checkpoint loaded from {path} (epoch {epoch}, loss {loss:.4f})")
    return model, optimizer, epoch, loss


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