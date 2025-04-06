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

class BaseConvBlock(nn.Module):
    """
    Basic convolutional block with configurable options and optional residual connection.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        dimension (int): Spatial dimension (1, 2, or 3)
        kernel_size (int): Size of the convolutional kernel
        stride (int): Stride of the convolution
        padding (int): Padding size
        use_batchnorm (bool): Whether to use batch normalization
        activation (str): Type of activation ('leaky_relu', 'relu', 'gelu', or 'none')
        dropout_rate (float): Dropout probability (0.0 to disable)
        leaky_slope (float): Slope for LeakyReLU
        bias (bool): Whether to use bias in convolution
        use_residual (bool): Whether to use residual connection
    """
    def __init__(self, in_channels, out_channels, dimension=2, kernel_size=3, stride=1, padding=1,
                 use_batchnorm=True, activation='leaky_relu', dropout_rate=0.0,
                 leaky_slope=0.1, bias=False, use_residual=False):
        super().__init__()
        
        ConvNd, _, _, BatchNormNd, DropoutNd, _ = getNdTools(dimension)
        self.use_residual = use_residual and (in_channels == out_channels) and (stride == 1)
        
        layers = [
            ConvNd(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        ]
        
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
    """
    Base U-Net architecture with flexible configuration and residual connections.
    
    Args:
        in_channels (int): Number of input channels
        num_filters (list): List of filter sizes for each layer
        dimension (int): Spatial dimension (1, 2, or 3)
        kernel_size (int): Size of convolutional kernels
        use_batchnorm (bool): Whether to use batch normalization
        activation (str): Type of activation function
        dropout_rate (float): Dropout probability
        leaky_slope (float): Slope for LeakyReLU
        bias (bool): Whether to use bias in convolutions
        use_residual (bool): Whether to use residual connections
    """
    def __init__(self, in_channels, num_filters=[64, 128, 256, 512], dimension=2,
                 kernel_size=3, use_batchnorm=True, activation='leaky_relu',
                 dropout_rate=0.0, leaky_slope=0.1, bias=False, use_residual=False):
        super().__init__()
        
        ConvNd, ConvTransposeNd, MaxPoolNd, _, _, _ = getNdTools(dimension)
        self.num_filters = num_filters
        self.pool = MaxPoolNd(kernel_size=2, stride=2)
        
        # Encoder
        self.downs = nn.ModuleList()
        current_channels = in_channels
        for filters in num_filters:
            self.downs.append(BaseConvBlock(
                current_channels, filters, dimension, kernel_size, 1, padding=1,
                use_batchnorm, activation, dropout_rate, leaky_slope, bias, use_residual
            ))
            current_channels = filters
            
        # Bottleneck
        self.bottleneck = BaseConvBlock(
            num_filters[-1], num_filters[-1]*2, dimension, kernel_size, 1, padding=1,
            use_batchnorm, activation, dropout_rate, leaky_slope, bias, use_residual
        )
        
        # Decoder
        self.ups = nn.ModuleList()
        for filters in reversed(num_filters):
            self.ups.append(nn.Sequential(
                ConvTransposeNd(filters*2, filters, kernel_size=2, stride=2),
                BaseConvBlock(filters*2, filters, dimension, kernel_size, 1, padding=1,
                            use_batchnorm, activation, dropout_rate, leaky_slope, bias, use_residual)
            ))
            
    def forward_features(self, x):
        """Extract features from the encoder path"""
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
    """
    Base LeNet architecture with flexible configuration and residual connections.
    
    Args:
        in_channels (int): Number of input channels
        num_filters (list): List of filter sizes for each layer
        dimension (int): Spatial dimension (1, 2, or 3)
        kernel_size (int): Size of convolutional kernels
        use_batchnorm (bool): Whether to use batch normalization
        activation (str): Type of activation function
        dropoutA_rate (float): Dropout probability
        leaky_slope (float): Slope for LeakyReLU
        bias (bool): Whether to use bias in convolutions
        use_residual (bool): Whether to use residual connections
    """
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
    """
    Configurable network head for different tasks.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        dimension (int): Spatial dimension (1, 2, or 3)
        task (str): Type of task ('regression', 'classification', 'segmentation')
        fc_layers (list): List of fully connected layer sizes
        dropout_rate (float): Dropout probability
        activation (str): Type of activation function
        leaky_slope (float): Slope for LeakyReLU
        bias (bool): Whether to use bias
    """
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
    Enhanced Multi-task U-Net architecture.
    
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
    """
    def __init__(self, in_channels, out_channels, dimension=2, num_filters=[64, 128, 256],
                 task='regression', use_batchnorm=True, activation='leaky_relu',
                 dropout_rate=0.0, leaky_slope=0.1, bias=False, fc_layers=[1024, 512],
                 extra_params_dim=0, use_residual=False):
        super().__init__()
        
        self.dimension = dimension
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
        Extract features from the encoder path of the U-Net.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, *spatial_dims)
            
        Returns:
            tuple: (bottleneck_features, skip_connections)
                - bottleneck_features (torch.Tensor): Features after the bottleneck
                - skip_connections (list): List of skip connection tensors from the encoder
        """
        bottleneck_features, skip_connections = self.base.forward_features(x)
        bottleneck_features = self.base.bottleneck(bottleneck_features)
        return bottleneck_features, skip_connections

class EMLeNet(nn.Module):
    """
    Enhanced Multi-task LeNet architecture.
    
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
    """
    def __init__(self, in_channels, out_channels, dimension=2, num_filters=[16, 32, 64],
                 task='regression', use_batchnorm=True, activation='leaky_relu',
                 dropout_rate=0.0, leaky_slope=0.1, bias=False, fc_layers=[1024, 512],
                 extra_params_dim=0, use_residual=False):
        super().__init__()
        
        self.dimension = dimension
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
        Extract features from the convolutional layers of the LeNet.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, *spatial_dims)
            
        Returns:
            torch.Tensor: Features after the convolutional layers
        """
        return self.base.forward_features(x)

# Example usage
if __name__ == "__main__":
    # 1D U-Net for regression
    unet_1d = EMUNet(
        in_channels=2,
        out_channels=1,
        dimension=1,
        num_filters=[32, 64],
        task='regression',
        extra_params_dim=3
    )
    
    # 2D LeNet for classification
    lenet_2d = EMLeNet(
        in_channels=1,
        out_channels=10,
        dimension=2,
        num_filters=[16, 32, 64],
        task='classification',
        fc_layers=[512, 256]
    )
    
    # 3D U-Net for segmentation
    unet_3d = EMUNet(
        in_channels=3,
        out_channels=4,
        dimension=3,
        num_filters=[32, 64, 128],
        task='segmentation',
        use_residual=True
    )
    
    # Test with random inputs
    x_1d = torch.randn(2, 2, 128)
    params = torch.randn(2, 3)
    out_1d = unet_1d(x_1d, params)
    features_1d, skip_1d = unet_1d.extract_features(x_1d)
    print(f"1D Regression output shape: {out_1d.shape}")
    print(f"1D Extracted bottleneck features shape: {features_1d.shape}")
    print(f"1D Skip connections: {[s.shape for s in skip_1d]}")
    
    x_2d = torch.randn(2, 1, 32, 32)
    out_2d = lenet_2d(x_2d)
    features_2d = lenet_2d.extract_features(x_2d)
    print(f"2D Classification output shape: {out_2d.shape}")
    print(f"2D Extracted features shape: {features_2d.shape}")
    
    x_3d = torch.randn(2, 3, 16, 16, 16)
    out_3d = unet_3d(x_3d)
    features_3d, skip_3d = unet_3d.extract_features(x_3d)
    print(f"3D Segmentation output shape: {out_3d.shape}")
    print(f"3D Extracted bottleneck features shape: {features_3d.shape}")
    print(f"3D Skip connections: {[s.shape for s in skip_3d]}")