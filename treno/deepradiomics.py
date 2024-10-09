import torch
import torch.nn as nn
import torch.nn.functional as F
import math
def output_size(input_size, kernel_size, stride, padding, dimension=3):
    """
    Calculate the output size of a 2D or 3D convolution layer.
    works, for 2D and 3D convolutions and pooling layers.
    
    Args:
    input_size (tuple): A tuple representing the size of the input tensor (height, width) for 2D
                        or (depth, height, width) for 3D.
    kernel_size (int or tuple): The size of the kernel in each dimension.
    stride (int or tuple): The stride in each dimension.
    padding (int or tuple): The padding in each dimension.
    dimension (int): Either 2 for 2D convolutions or 3 for 3D convolutions.

    Returns:
    tuple: The size of the output tensor.
    """
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size,) * dimension
    if isinstance(stride, int):
        stride = (stride,) * dimension
    if isinstance(padding, int):
        padding = (padding,) * dimension
    
    # Unpack input size, kernel size, stride, and padding
    if dimension == 3:
        depth, height, width = input_size
        kd, kh, kw = kernel_size
        sd, sh, sw = stride
        pd, ph, pw = padding

        # Apply the formula to calculate output dimensions
        output_depth = math.floor((depth - kd + 2 * pd) / sd) + 1
        output_height = math.floor((height - kh + 2 * ph) / sh) + 1
        output_width = math.floor((width - kw + 2 * pw) / sw) + 1

        return (output_depth, output_height, output_width)
    
    elif dimension == 2:
        height, width = input_size
        kh, kw = kernel_size
        sh, sw = stride
        ph, pw = padding

        # Apply the formula to calculate output dimensions
        output_height = math.floor((height - kh + 2 * ph) / sh) + 1
        output_width = math.floor((width - kw + 2 * pw) / sw) + 1

        return (output_height, output_width)

    else:
        
        raise ValueError("Only 2D or 3D convolutions are supported")
    

def getNdTools(dimension):
    if dimension == 1:
        return nn.Conv1d, nn.ConvTranspose1d, nn.AvgPool1d, nn.BatchNorm1d, nn.Dropout1d,nn.ReflectionPad1d
    elif dimension == 2:
        return nn.Conv2d, nn.ConvTranspose2d,nn.AvgPool2d, nn.BatchNorm2d, nn.Dropout2d,nn.ReflectionPad2d
    elif dimension == 3:
        return nn.Conv3d, nn.ConvTranspose3d, nn.AvgPool3d, nn.BatchNorm3d, nn.Dropout3d,nn.ReflectionPad3d
    else:
        raise ValueError("Only 1, 2, or 3 dimensions are supported")


# class HybridNormalizationPredictor(nn.Module):
#     def __init__(self, dimension,metadata_input_size):
#         super().__init__()
        
#         # Convolutional layers for the image input
#         C, U, P, B, D = getNdTools(dimension)
        
#         # Define convolutional and pooling layers according to parameters
#         conv_params = [{'out_channels': 16, 'kernel_size': 3, 'stride': 1, 'padding': 1}, 
#                        {'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'padding': 1}]
#         pool_params = [{'kernel_size': 2, 'stride': 2}, {'kernel_size': 2, 'stride': 2}]
        
#         self.convs = nn.ModuleList([C(in_channels=conv_params[i]['out_channels'] if i > 0 else 1, **params) 
#                                     for i, params in enumerate(conv_params)])
#         self.pools = nn.ModuleList([P(**params) for params in pool_params])
        
#         # Calculate the flattened size after the conv layers
#         volume_size = 64
#         output_channels = conv_params[-1]['out_channels']

#         self.flatten_size = output_channels * calculate_flattened_size(volume_size, conv_params, pool_params)**3
        
#         # Fully connected layers for the metadata (TE, TR, sequence type, etc.)
#         self.fc_metadata_1 = nn.Linear(metadata_input_size, 16)
#         self.fc_metadata_2 = nn.Linear(16, 8)
        
#         # Combined fully connected layers (image + metadata)
#         self.fc_combined_1 = nn.Linear(self.flatten_size + 8, 32)
#         self.fc_combined_2 = nn.Linear(32, 16)
#         self.fc_output = nn.Linear(16, 1)  # Output normalization value
        
#         # Activation function
#         self.relu = nn.SELU()
        
    
#     def forward(self, x, metadata):
#         for conv, pool in zip(self.convs, self.pools):
#             x = pool(self.relu(conv(x)))
#         x = x.view(x.size(0), -1)  # Flatten the tensor

#         # Metadata processing through fully connected layers
#         x_metadata = self.relu(self.fc_metadata_1(metadata))
#         x_metadata = self.relu(self.fc_metadata_2(x_metadata))
        
#         # Concatenate image features and metadata features
#         x_combined = torch.cat((x, x_metadata), dim=1)
        
#         # Combined fully connected layers
#         x_combined = self.relu(self.fc_combined_1(x_combined))
#         x_combined = self.relu(self.fc_combined_2(x_combined))
        
#         # Output layer (single normalization value)
#         output = self.fc_output(x_combined)
#         return output
    
def tensor_memory_size(tensor):
    return tensor.element_size() * tensor.nelement()/1024/1024

from functools import reduce
class DeepRadioNet(nn.Module):
    """
    DeepRadioNet is a deep learning model for MRI image classification.
    It consists of five convolutional layers followed by three fully connected layers.
    #https://www.nature.com/articles/s41598-017-10649-8/figures/2
    
    Author: eros.montin@gmail.com
    Date: 2024-10-08
    
    

    Args:
        nn (_type_): _description_
    """
    def __init__(self,dimension,image_size,nchan=1):
    
        super().__init__()
        # Convolutional layers
        C, U, P, B, D,A = getNdTools(dimension)
        self.conv1 = C(in_channels=nchan, out_channels=96, kernel_size=7, stride=2, padding=0)  # 7x7x7 kernel
        self.conv2 = C(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=1)  # 5x5x5 kernel
        self.conv3 = C(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)  # 3x3x3 kernel
        self.conv4 = C(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)  # 3x3x3 kernel
        self.conv5 = C(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)  # 3x3x3 kernel

        # Max pooling
        self.pool3 = P(kernel_size=3, stride=3)
        self.pool2 = P(kernel_size=2, stride=2)
        self.lrn = nn.LocalResponseNorm(5)
        self.dropout = nn.Dropout(0.5)
        
        # Calculate the size of the output tensor after the conv layers
        output_size1 = output_size(image_size, 7, 2, 0, dimension)
        # Calculate the size of the output tensor after the first pooling layer x3
        output_size2 = output_size(output_size1, 3, 3, 0, dimension)

        # Calculate the size of the output tensor after the second conv layer
        output_size3 = output_size(output_size2, 5, 1, 1, dimension)

        # Calculate the size of the output tensor after the second pooling layer x2
        output_size4 = output_size(output_size3, 2, 2, 0, dimension)

        # Calculate the size of the output tensor after the third conv layer
        output_size5 = output_size(output_size4, 3, 1, 1, dimension)

        # Calculate the size of the output tensor after the fourth conv layer
        output_size6 = output_size(output_size5, 3, 1, 1, dimension)
    
        # Calculate the size of the output tensor after the fifth conv layer
        output_size7 = output_size(output_size6, 3, 1, 1, dimension)
    
        # Calculate the size of the output tensor after the third pooling layer x3
        output_size8 = output_size(output_size7, 3, 3, 0, dimension)
    
        FC=reduce(lambda x, y: x*y, output_size8)
        
        self.fc1 = nn.Linear(FC*512, 4096)

        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 8192)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        # Forward pass through conv layers with pooling
        x = self.pool3(self.conv1(x))
        x=self.lrn(x)
        x = self.pool2(self.conv2(x))
        x=self.lrn(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool3(self.conv5(x))
        

        # Flatten before fully connected layers
        x = torch.flatten(x, 1)  # Flatten all dimensions except batch
        
        # Fully connected layers
        x = self.dropout(self.fc1(x))
        x = self.dropout(self.fc2(x))
        x = self.fc3(x)  # Output layer
        x = self.softmax(x)
        return x

class DeepRadioClassifier(nn.Module):
    """
    Combines DeepRadioNet with a classification layer for final output.
    """
    def __init__(self, num_classes=3):
        super().__init__()
        self.classifier = nn.Linear(8192, num_classes)
    
    def forward(self, x):
        # Pass through the final classification layer
        x = self.classifier(x)
        return x
    
# import numpy as np
# import pandas as pd
# from scipy.stats import zscore
# from sklearn.metrics import pairwise_distances
# from statsmodels.stats.inter_rater import fleiss_kappa

# def evaluate_feature_robustness(features):
#     """
#     Evaluate the robustness of features using ICC.
#     Features with ICC ≥ 0.85 are considered robust.
#     """
#     # This is a placeholder for your ICC calculation logic
#     robust_features = []
#     for feature in features.columns:
#         # Calculate ICC here based on test-retest and inter-rater analysis
#         icc_value = calculate_icc(features[feature])  # Define calculate_icc
#         if icc_value >= 0.85:
#             robust_features.append(feature)
#     return features[robust_features]

# def calculate_mad(features):
#     """
#     Calculate Median Absolute Deviation (MAD) for each feature.
#     Discard features with MAD equal to zero.
#     """
#     mad_values = features.mad()
#     return features.loc[:, mad_values != 0]

# def assess_prognostic_value(features, targets):
#     """
#     Assess prognostic value using C-index.
#     Features with C-index ≥ 0.580 are retained.
#     """
#     prognostic_features = []
#     for feature in features.columns:
#         c_index = calculate_c_index(features[feature], targets)  # Define calculate_c_index
#         if c_index >= 0.580:
#             prognostic_features.append(feature)
#     return features[prognostic_features]

# def remove_highly_correlated_features(features):
#     """
#     Remove highly correlated features (correlation coefficient ≥ 0.90).
#     Retain the more prognostic feature from each correlated pair.
#     """
#     corr_matrix = features.corr().abs()
#     to_drop = set()
    
#     for i in range(len(corr_matrix.columns)):
#         for j in range(i):
#             if corr_matrix.iloc[i, j] >= 0.90:
#                 feature_i = corr_matrix.columns[i]
#                 feature_j = corr_matrix.columns[j]
#                 # Here you would compare prognostic values and retain the better one
#                 if feature_j in to_drop:
#                     continue
#                 if feature_j not in to_drop:  # Compare the features to determine which to drop
#                     # Assume we have some function to compare prognostic values
#                     if not compare_prognostic_value(feature_i, feature_j):
#                         to_drop.add(feature_j)
#                     else:
#                         to_drop.add(feature_i)
#     return features.drop(columns=to_drop)

# def feature_selection_pipeline(features, targets):
#     """
#     Execute the complete feature selection pipeline.
#     """
#     # Step 1: Evaluate feature robustness
#     robust_features = evaluate_feature_robustness(features)

#     # Step 2: Calculate MAD and remove non-informative features
#     mad_features = calculate_mad(robust_features)

#     # Step 3: Assess prognostic value
#     prognostic_features = assess_prognostic_value(mad_features, targets)

#     # Step 4: Remove highly correlated features
#     final_features = remove_highly_correlated_features(prognostic_features)

#     return final_features


if __name__ == "__main__":

   	# "MagneticFieldStrength": 3,
	# "ImagingFrequency": 123.262284,
	# "Manufacturer": "Siemens",
	# "ManufacturersModelName": "Prisma_fit",
	# "PatientPosition": "FFS",
	# "MRAcquisitionType": "2D",
	# "ScanningSequence": "SE",
	# "SAR": 1.42141,
	# "NumberOfAverages": 2,
	# "EchoTime": 0.038,
	# "RepetitionTime": 3.8,
	# "SpoilingState": true,
	# "FlipAngle": 120,
	# "PixelBandwidth": 250,
	# "DwellTime": 6.2e-06,
    import numpy as np
    import torch

    # Define the size of the 3D image
    image_size = (160,320,120)  # Replace with the actual size expected by your network

    # Generate a random 3D array of the specified size
    image = np.random.rand(*image_size).astype(np.float32)

    # Convert the numpy array to a PyTorch tensor
    image_tensor = torch.from_numpy(image)

    # Add an extra dimension for the batch size and move the tensor to the correct device
    image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)

    model=DeepRadioNet(3,image_size,nchan=1)
    # Pass the tensor to the network
    output = model(image_tensor)
    print(output.shape)
    
    
#     # Example usage
# # Assuming `data` is a DataFrame with features and `target` is the outcome variable
# data = pd.DataFrame(np.random.rand(100, 1403))  # Example features
# target = np.random.randint(0, 2, size=(100,))  # Example binary outcome

# # Normalize features as z-scores
# data_z = data.apply(zscore)

# # Run feature selection pipeline
# selected_features = feature_selection_pipeline(data_z, target)
# print(f"Number of selected features: {selected_features.shape[1]}")
