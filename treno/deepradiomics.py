import torch
import torch.nn as nn
import numpy as np

def getNdTools(dimension):
    """Returns appropriate PyTorch modules based on the specified dimension."""
    if dimension == 1:
        return nn.Conv1d, nn.ConvTranspose1d, nn.MaxPool1d, nn.BatchNorm1d, nn.Dropout, nn.ReflectionPad1d
    elif dimension == 2:
        return nn.Conv2d, nn.ConvTranspose2d, nn.MaxPool2d, nn.BatchNorm2d, nn.Dropout2d, nn.ReflectionPad2d
    elif dimension == 3:
        return nn.Conv3d, nn.ConvTranspose3d, nn.MaxPool3d, nn.BatchNorm3d, nn.Dropout3d, nn.ReflectionPad3d
    else:
        raise ValueError("Only 1, 2, or 3 dimensions are supported")

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
    # Normalize x to [0, 1] range for consistent binning
    x_min, x_max = torch.min(x), torch.max(x)
    x_norm = (x - x_min) / (x_max - x_min + 1e-6) if x_max > x_min else x
    
    energy = torch.sum(x**2).item()
    total_energy = torch.sum(x).item()
    h = torch.histogram(x_norm.flatten(), bins=num_bins, density=True)[0]
    h = h[h > 1e-5]
    entropy = -torch.sum(h * torch.log(h + 1e-6)).item()
    minimum = torch.min(x).item()
    tenth_percentile = torch.quantile(x.flatten(), 0.1).item()
    twenty_fifth_percentile = torch.quantile(x.flatten(), 0.25).item()
    seventy_fifth_percentile = torch.quantile(x.flatten(), 0.75).item()
    ninetieth_percentile = torch.quantile(x.flatten(), 0.9).item()
    maximum = torch.max(x).item()
    mean = torch.mean(x).item()
    median = torch.median(x).item()
    mode = h.argmax().item() / (num_bins - 1)  # Scale mode to [0, 1]
    interquartile_range = seventy_fifth_percentile - twenty_fifth_percentile
    range_ = maximum - minimum
    mad = torch.mean(torch.abs(x - mean)).item()
    mad_median = torch.mean(torch.abs(x - median)).item()
    rms = torch.sqrt(torch.mean(x**2)).item()
    std_dev = torch.std(x).item()
    variance = torch.var(x).item()
    skewness = calculate_skewness_torch(x).item()
    kurtosis = calculate_kurtosis_torch(x).item()
    unique_elements = torch.unique(x)
    uniformity = len(unique_elements) / x.numel()
    cv = std_dev / (mean + 1e-6)
    diff_entropy = -torch.sum(torch.diff(x.flatten()) * torch.log(torch.abs(torch.diff(x.flatten())) + 1e-6)).item()

    return torch.tensor([
        energy, cv, total_energy, entropy, minimum, tenth_percentile, twenty_fifth_percentile,
        seventy_fifth_percentile, ninetieth_percentile, maximum, mean, median, mode,
        interquartile_range, range_, mad, mad_median, rms, std_dev, skewness, kurtosis,
        variance, uniformity, diff_entropy
    ])

def calculate_simple_glcm_features(x, radii=[1], dimension=2):
    """Calculate simplified GLCM-like features for a tensor across multiple radii."""
    glcm_features = []
    for radius in radii:
        if dimension == 1:
            x_shift = torch.roll(x, shifts=radius, dims=0)
            x_shift[:radius] = 0  # Padding
        elif dimension == 2:
            x_shift = torch.roll(x, shifts=radius, dims=1)
            x_shift[:, :radius] = 0
        else:  # 3D
            x_shift = torch.roll(x, shifts=radius, dims=2)
            x_shift[:, :, :radius] = 0
        
        contrast = torch.mean((x - x_shift) ** 2).item()
        energy = torch.sum(x**2).item()
        homogeneity = torch.mean(1 / (1 + torch.abs(x - x_shift))).item()
        glcm_features.extend([contrast, energy, homogeneity])
    
    return torch.tensor(glcm_features)

class BaseConvBlock(nn.Module):
    # [Unchanged]
    def __init__(self, in_channels, out_channels, dimension=2, kernel_size=3, stride=1, padding=1,
                 use_batchnorm=True, activation='leaky_relu', dropout_rate=0.0,
                 leaky_slope=0.1, bias=False, use_residual=False):
        super().__init__()
        ConvNd, _, _, BatchNormNd, DropoutNd, _ = getNdTools(dimension)
        self.use_residual = use_residual and (in_channels == out_channels) and (stride == 1)
        
        layers = [ConvNd(in_channels, out_channels, kernel_size, stride, padding, bias=bias)]
        if use_batchnorm:
            layers.append(BatchNormNd(out_channels))
            
        if activation == 'leaky_relu':
            self.act = nn.LeakyReLU(leaky_slope, inplace=True)
        elif activation == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif activation == 'gelu':
            self.act = nn.GELU()
        else:
            self.act = nn.Identity()
            
        layers.append(self.act)
        if dropout_rate > 0:
            layers.append(DropoutNd(dropout_rate))
            
        self.block = nn.Sequential(*layers)
        
    def forward(self, x):
        out = self.block(x)
        if self.use_residual:
            out = out + x
        return out

class UNetBase(nn.Module):
    # [Unchanged]
    def __init__(self, in_channels, num_filters=[64, 128, 256, 512], dimension=2,
                 kernel_size=3, use_batchnorm=True, activation='leaky_relu',
                 dropout_rate=0.0, leaky_slope=0.1, bias=False, use_residual=False):
        super().__init__()
        ConvNd, ConvTransposeNd, MaxPoolNd, _, _, _ = getNdTools(dimension)
        self.num_filters = num_filters
        self.pool = MaxPoolNd(kernel_size=2, stride=2)
        
        self.downs = nn.ModuleList()
        current_channels = in_channels
        for filters in num_filters:
            self.downs.append(BaseConvBlock(
                current_channels, filters, dimension, kernel_size, 1, padding=1,
                use_batchnorm, activation, dropout_rate, leaky_slope, bias, use_residual
            ))
            current_channels = filters
            
        self.bottleneck = BaseConvBlock(
            num_filters[-1], num_filters[-1]*2, dimension, kernel_size, 1, padding=1,
            use_batchnorm, activation, dropout_rate, leaky_slope, bias, use_residual
        )
        
        self.ups = nn.ModuleList()
        for filters in reversed(num_filters):
            self.ups.append(nn.Sequential(
                ConvTransposeNd(filters*2, filters, kernel_size=2, stride=2),
                BaseConvBlock(filters*2, filters, dimension, kernel_size, 1, padding=1,
                            use_batchnorm, activation, dropout_rate, leaky_slope, bias, use_residual)
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
                x = nn.functional.interpolate(x, size=skip.shape[2:], mode=mode, align_corners=False)
            x = torch.cat([skip, x], dim=1)
            x = up[1](x)
        return x

class LeNetBase(nn.Module):
    # [Unchanged]
    def __init__(self, in_channels, num_filters=[16, 32, 64], dimension=2,
                 kernel_size=3, use_batchnorm=True, activation='leaky_relu',
                 dropout_rate=0.0, leaky_slope=0.1, bias=False, use_residual=False):
        super().__init__()
        _, _, MaxPoolNd, _, _, _ = getNdTools(dimension)
        self.num_filters = num_filters
        self.pool = MaxPoolNd(kernel_size=2, stride=2)
        
        self.convs = nn.ModuleList()
        current_channels = in_channels
        for filters in num_filters[:-1]:
            self.convs.append(BaseConvBlock(
                current_channels, filters, dimension, kernel_size, 1, padding=1,
                use_batchnorm, activation, dropout_rate, leaky_slope, bias, use_residual
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
    # [Unchanged]
    def __init__(self, in_channels, out_channels, dimension=2, task='regression',
                 fc_layers=[1024, 512], dropout_rate=0.0, activation='leaky_relu',
                 leaky_slope=0.1, bias=False):
        super().__init__()
        ConvNd, _, _, _, _, _ = getNdTools(dimension)
        self.task = task.lower()
        self.fc_layers = nn.ModuleList()
        
        if self.task == 'segmentation':
            self.head = ConvNd(in_channels, out_channels, kernel_size=1, bias=bias)
        else:
            current_channels = in_channels
            for fc_size in fc_layers:
                self.fc_layers.append(nn.Sequential(
                    nn.Linear(current_channels, fc_size, bias=bias),
                    self._get_activation(activation, leaky_slope),
                    nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
                ))
                current_channels = fc_size
            self.fc_layers.append(nn.Linear(current_channels, out_channels, bias=bias))
            
    def _get_activation(self, activation, leaky_slope):
        if activation == 'leaky_relu':
            return nn.LeakyReLU(leaky_slope, inplace=True)
        elif activation == 'relu':
            return nn.ReLU(inplace=True)
        elif activation == 'gelu':
            return nn.GELU()
        return nn.Identity()
    
    def forward(self, x):
        if self.task == 'segmentation':
            x = self.head(x)
            return nn.Softmax(dim=1)(x)
        else:
            x = torch.flatten(x, 1)
            for layer in self.fc_layers:
                x = layer(x)
            return torch.sigmoid(x) if self.task == 'classification' else x

class EMUNet(nn.Module):
    """
    Enhanced Multi-task U-Net architecture with optional radiomics features.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        dimension (int): Spatial dimension (1, 2, or 3)
        num_filters (list): List of filter sizes for each layer
        task (str): Type of task ('regression', 'classification', 'segmentation')
        use_batchnorm (bool): Whether to use batch normalization
        activation (str): Type of activation function
        dropout_rate (float): Dropout probability
        leaky_slope (float): Slope for LeakyReLU
        bias (bool): Whether to use bias
        fc_layers (list): List of fully connected layer sizes (for regression/classification)
        extra_params_dim (int): Dimension of optional extra parameters
        use_residual (bool): Whether to use residual connections
        use_radiomics (bool): Whether to compute radiomics features during extraction
        num_bins (int): Number of bins for histogram-based features
        radii (list): List of radii for textural (GLCM) features
    """
    def __init__(self, in_channels, out_channels, dimension=2, num_filters=[64, 128, 256],
                 task='regression', use_batchnorm=True, activation='leaky_relu',
                 dropout_rate=0.0, leaky_slope=0.1, bias=False, fc_layers=[1024, 512],
                 extra_params_dim=0, use_residual=False, use_radiomics=False,
                 num_bins=256, radii=[1]):
        super().__init__()
        
        self.dimension = dimension
        self.use_radiomics = use_radiomics
        self.num_bins = num_bins
        self.radii = radii
        self.base = UNetBase(
            in_channels, num_filters, dimension, 3, use_batchnorm,
            activation, dropout_rate, leaky_slope, bias, use_residual
        )
        
        self.extra_params_dim = extra_params_dim
        if extra_params_dim > 0:
            self.param_fc = nn.Linear(extra_params_dim, num_filters[0])
            
        self.head = NetworkHead(
            num_filters[0], out_channels, dimension, task, fc_layers,
            dropout_rate, activation, leaky_slope, bias
        )
        
    def forward(self, x, extra_params=None):
        x = self.base(x)
        
        if extra_params is not None and self.extra_params_dim > 0:
            params = self.param_fc(extra_params)
            params = params.view(x.shape[0], x.shape[1], *[1]*self.dimension)
            x = x + params
                
        return self.head(x)
    
    def extract_features(self, x):
        """
        Extract features from the encoder path of the U-Net, optionally including radiomics features.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, *spatial_dims)
            
        Returns:
            tuple: (bottleneck_features, skip_connections, stats_features)
                - bottleneck_features (torch.Tensor): Features after the bottleneck
                - skip_connections (list): List of skip connection tensors from the encoder
                - stats_features (torch.Tensor or None): Radiomics features (FOS + GLCM) per batch/channel if use_radiomics=True
        """
        bottleneck_features, skip_connections = self.base.forward_features(x)
        bottleneck_features = self.base.bottleneck(bottleneck_features)
        
        if self.use_radiomics:
            stats_features = torch.tensor([])
            for i in range(x.shape[0]):  # For each batch
                channelfeatures = torch.tensor([])
                for j in range(x.shape[1]):  # For each channel
                    fos = calculate_fos_features(x[i, j], num_bins=self.num_bins)
                    glcm = calculate_simple_glcm_features(x[i, j], radii=self.radii, dimension=self.dimension)
                    combined = torch.cat((fos, glcm))
                    combined = combined / (torch.max(torch.abs(combined)) + 1e-6)  # Normalize
                    channelfeatures = torch.cat((channelfeatures, combined))
                if i == 0:
                    stats_features = channelfeatures
                else:
                    stats_features = torch.vstack((stats_features, channelfeatures))
        else:
            stats_features = None
            
        return bottleneck_features, skip_connections, stats_features

class EMLeNet(nn.Module):
    """
    Enhanced Multi-task LeNet architecture with optional radiomics features.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        dimension (int): Spatial dimension (1, 2, or 3)
        num_filters (list): List of filter sizes for each layer
        task (str): Type of task ('regression', 'classification', 'segmentation')
        use_batchnorm (bool): Whether to use batch normalization
        activation (str): Type of activation function
        dropout_rate (float): Dropout probability
        leaky_slope (float): Slope for LeakyReLU
        bias (bool): Whether to use bias
        fc_layers (list): List of fully connected layer sizes (for regression/classification)
        extra_params_dim (int): Dimension of optional extra parameters
        use_residual (bool): Whether to use residual connections
        use_radiomics (bool): Whether to compute radiomics features during extraction
        num_bins (int): Number of bins for histogram-based features
        radii (list): List of radii for textural (GLCM) features
    """
    def __init__(self, in_channels, out_channels, dimension=2, num_filters=[16, 32, 64],
                 task='regression', use_batchnorm=True, activation='leaky_relu',
                 dropout_rate=0.0, leaky_slope=0.1, bias=False, fc_layers=[1024, 512],
                 extra_params_dim=0, use_residual=False, use_radiomics=False,
                 num_bins=256, radii=[1]):
        super().__init__()
        
        self.dimension = dimension
        self.use_radiomics = use_radiomics
        self.num_bins = num_bins
        self.radii = radii
        self.base = LeNetBase(
            in_channels, num_filters, dimension, 3, use_batchnorm,
            activation, dropout_rate, leaky_slope, bias, use_residual
        )
        
        self.extra_params_dim = extra_params_dim
        if extra_params_dim > 0:
            self.param_fc = nn.Linear(extra_params_dim, num_filters[-2])
            
        self.head = NetworkHead(
            num_filters[-2], out_channels, dimension, task, fc_layers,
            dropout_rate, activation, leaky_slope, bias
        )
        
    def forward(self, x, extra_params=None):
        x = self.base(x)
        
        if extra_params is not None and self.extra_params_dim > 0:
            params = self.param_fc(extra_params)
            params = params.view(x.shape[0], -1, *[1]*self.dimension)
            x = torch.cat([x, params], dim=1)
                
        return self.head(x)
    
    def extract_features(self, x):
        """
        Extract features from the convolutional layers of the LeNet, optionally including radiomics features.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, *spatial_dims)
            
        Returns:
            tuple: (conv_features, stats_features)
                - conv_features (torch.Tensor): Features after the convolutional layers
                - stats_features (torch.Tensor or None): Radiomics features (FOS + GLCM) per batch/channel if use_radiomics=True
        """
        conv_features = self.base.forward_features(x)
        
        if self.use_radiomics:
            stats_features = torch.tensor([])
            for i in range(x.shape[0]):  # For each batch
                channelfeatures = torch.tensor([])
                for j in range(x.shape[1]):  # For each channel
                    fos = calculate_fos_features(x[i, j], num_bins=self.num_bins)
                    glcm = calculate_simple_glcm_features(x[i, j], radii=self.radii, dimension=self.dimension)
                    combined = torch.cat((fos, glcm))
                    combined = combined / (torch.max(torch.abs(combined)) + 1e-6)  # Normalize
                    channelfeatures = torch.cat((channelfeatures, combined))
                if i == 0:
                    stats_features = channelfeatures
                else:
                    stats_features = torch.vstack((stats_features, channelfeatures))
        else:
            stats_features = None
            
        return conv_features, stats_features

# Example usage
if __name__ == "__main__":
    # 1D U-Net with custom radiomics
    unet_1d = EMUNet(
        in_channels=2, out_channels=1, dimension=1, num_filters=[32, 64],
        task='regression', extra_params_dim=3, use_radiomics=True,
        num_bins=128, radii=[1, 2]
    )
    
    # 2D LeNet without radiomics
    lenet_2d = EMLeNet(
        in_channels=1, out_channels=10, dimension=2, num_filters=[16, 32, 64],
        task='classification', fc_layers=[512, 256], use_radiomics=False
    )
    
    # 3D U-Net with custom radiomics
    unet_3d = EMUNet(
        in_channels=3, out_channels=4, dimension=3, num_filters=[32, 64, 128],
        task='segmentation', use_residual=True, use_radiomics=True,
        num_bins=64, radii=[1, 3, 5]
    )
    
    # Test with random inputs
    x_1d = torch.randn(2, 2, 128)
    params = torch.randn(2, 3)
    out_1d = unet_1d(x_1d, params)
    features_1d, skip_1d, stats_1d = unet_1d.extract_features(x_1d)
    print(f"1D Regression output shape: {out_1d.shape}")
    print(f"1D Extracted bottleneck features shape: {features_1d.shape}")
    print(f"1D Skip connections: {[s.shape for s in skip_1d]}")
    print(f"1D Statistical features shape: {stats_1d.shape if stats_1d is not None else 'None'}")
    
    x_2d = torch.randn(2, 1, 32, 32)
    out_2d = lenet_2d(x_2d)
    features_2d, stats_2d = lenet_2d.extract_features(x_2d)
    print(f"2D Classification output shape: {out_2d.shape}")
    print(f"2D Extracted features shape: {features_2d.shape}")
    print(f"2D Statistical features shape: {stats_2d.shape if stats_2d is not None else 'None'}")
    
    x_3d = torch.randn(2, 3, 16, 16, 16)
    out_3d = unet_3d(x_3d)
    features_3d, skip_3d, stats_3d = unet_3d.extract_features(x_3d)
    print(f"3D Segmentation output shape: {out_3d.shape}")
    print(f"3D Extracted bottleneck features shape: {features_3d.shape}")
    print(f"3D Skip connections: {[s.shape for s in skip_3d]}")
    print(f"3D Statistical features shape: {stats_3d.shape if stats_3d is not None else 'None'}")