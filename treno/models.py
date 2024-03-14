# unet.py file

import numpy as np
from torch import nn
import torch
import torchvision.transforms.functional as TF
# https://pyimagesearch.com/2021/11/08/u-net-training-image-segmentation-models-in-pytorch/
# https://amaarora.github.io/2020/09/13/unet.html
# https://idiomaticprogrammers.com/post/unet-architecture/


class SingleBox(nn.Module):
    """
    A module that represents a single box in a neural network model.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        conv (nn.Module): Convolutional layer module.
        batch (nn.Module, optional): Batch normalization module. Defaults to None.
        relu (nn.Module): ReLU activation module.
        kernel_size (int or tuple): Size of the convolutional kernel.
        strides (int or tuple): Strides for the convolution operation.
        padding (int or tuple): Padding for the convolution operation.
        bias (bool, optional): Whether to include bias in the convolutional layer. Defaults to False.
        leaky_relu (float, optional): Negative slope for the LeakyReLU activation. Defaults to 0.1.
    """

    def __init__(self, in_channels, out_channels, conv, batch, relu, kernel_size, strides, padding, bias=False, leaky_relu=0.1):
        super().__init__()
        if batch is not None:
            # Your code here
            pass
            self.core = nn.Sequential(
                conv(in_channels, out_channels, kernel_size,
                     strides, padding, bias=bias),
                batch(num_features=out_channels),
                relu(leaky_relu, inplace=True)
            )
        else:
            self.core = nn.Sequential(
                conv(in_channels, out_channels, kernel_size,
                     strides, padding, bias=bias),
                relu(leaky_relu, inplace=True)
            )

    def forward(self, x):
        return self.core(x)


class DoubleBox(SingleBox):
    """
    A class representing a double box in a neural network model.

    Inherits from the SingleBox class and extends it by applying two SingleBox layers sequentially.
    in the and then a box normalization and a relu activation.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        conv (nn.Module): Convolutional layer module.
        batch (nn.Module): Batch normalization layer module.
        relu (nn.Module): ReLU activation layer module.
        kernel_size (int or tuple): Size of the convolutional kernel.
        strides (int or tuple): Stride of the convolution.
        padding (int or tuple): Padding added to the input.
        bias (bool, optional): Whether to include a bias term in the convolutional layer. Defaults to False.
        leaky_relu (float, optional): Negative slope coefficient for LeakyReLU activation. Defaults to 0.1.
    """

    def __init__(self, in_channels, out_channels, conv, batch, relu, kernel_size, strides, padding, bias=False, leaky_relu=0.1):
        super().__init__(in_channels, out_channels, conv, batch,
                         relu, kernel_size, strides, padding, bias, leaky_relu)
        self.conv = nn.Sequential(
            SingleBox(in_channels, out_channels, conv, batch, relu,
                      kernel_size, strides, padding, bias, leaky_relu),
            SingleBox(in_channels, out_channels, conv, batch, relu,
                      kernel_size, strides, padding, bias, leaky_relu),
        )

class DoubleBoxv2(SingleBox):
    """
    A class representing a double box in a neural network model.
    the right version of the double box

    Inherits from the SingleBox class and extends it by applying two SingleBox layers sequentially.
    in the and then a box normalization and a relu activation.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        conv (nn.Module): Convolutional layer module.
        batch (nn.Module): Batch normalization layer module.
        relu (nn.Module): ReLU activation layer module.
        kernel_size (int or tuple): Size of the convolutional kernel.
        strides (int or tuple): Stride of the convolution.
        padding (int or tuple): Padding added to the input.
        bias (bool, optional): Whether to include a bias term in the convolutional layer. Defaults to False.
        leaky_relu (float, optional): Negative slope coefficient for LeakyReLU activation. Defaults to 0.1.
    """

    def __init__(self, in_channels, out_channels, conv, batch, relu, kernel_size, strides, padding, bias=False, leaky_relu=0.1):
        super().__init__(in_channels, out_channels, conv, batch,
                         relu, kernel_size, strides, padding, bias, leaky_relu)
        self.conv = nn.Sequential(
            SingleBox(in_channels, out_channels, conv, batch, relu,
                      kernel_size, strides, padding, bias, leaky_relu),
            SingleBox(out_channels, out_channels, conv, batch, relu,
                      kernel_size, strides, padding, bias, leaky_relu),
        )
    
class ResidualBox(SingleBox):
    """
    ResidualBox class represents a residual block in a neural network model.
    It inherits from the SingleBox class.
    and it has a shortcut layer that is added to the output of the SingleBox layer.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        conv (torch.nn.Module): Convolutional layer module.
        batch (torch.nn.Module): Batch normalization module.
        relu (torch.nn.Module): ReLU activation module.
        kernel_size (int or tuple): Size of the convolutional kernel.
        strides (int or tuple): Stride of the convolution.
        padding (int or tuple): Padding added to the input.
        bias (bool, optional): Whether to include a bias term in the convolutional layer. Defaults to False.
        leaky_relu (float, optional): Negative slope of the LeakyReLU activation function. Defaults to 0.1.
    """

    def __init__(self, in_channels, out_channels, conv, batch, relu, kernel_size, strides, padding, bias=False, leaky_relu=0.1):
        super().__init__(in_channels, out_channels, conv, batch,
                         relu, kernel_size, strides, padding, bias, leaky_relu)
        self.relu=relu
        self.leaky_relu=leaky_relu
        self.shortcut=nn.Sequential(
            conv(in_channels, out_channels, kernel_size, strides, padding, bias=bias),
            batch(num_features=out_channels),
        )
        
    def forward(self, x):
        o=super().forward(x)
        o+=self.shortcut(x)
        R=self.relu(self.leaky_relu,inplace=True)
        return R(o)
class DoubleBoxResidual(ResidualBox):
    """
    A class representing a double residual box in a neural network model.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        conv: Convolutional layer function.
        batch: Batch normalization layer function.
        relu: ReLU activation function.
        kernel_size: Size of the convolutional kernel.
        strides: Stride value for the convolution.
        padding: Padding value for the convolution.
        bias (bool, optional): Whether to include bias in the convolutional layer. Defaults to False.
        leaky_relu (float, optional): Negative slope for the LeakyReLU activation function. Defaults to 0.1.
    """

    def __init__(self, in_channels, out_channels, conv, batch, relu, kernel_size, strides, padding, bias=False, leaky_relu=0.1):
        super().__init__(in_channels, out_channels, conv, batch,
                         relu, kernel_size, strides, padding, bias, leaky_relu)

        self.core = nn.Sequential(
            SingleBox(in_channels, out_channels, conv, batch, relu,
                      kernel_size, strides, padding, bias, leaky_relu),
            SingleBox(out_channels, out_channels, conv, batch, relu,
                      kernel_size, strides, padding, bias, leaky_relu),
        )
class DoubleBoxParallel(nn.Module):
    
    def __init__(self, in_channels, out_channels, conv, batch, relu, kernel_size, strides, padding, bias=False, leaky_relu=0.1):
        super().__init__(in_channels, out_channels, conv, batch,
                         relu, kernel_size, strides, padding, bias, leaky_relu)
        # cat the output of the two convolutions
        self.relu=relu
        self.leaky_relu=leaky_relu
        self.conv0=SingleBox(in_channels, out_channels, conv, batch, relu,
                      kernel_size, strides, padding, bias, leaky_relu)
        self.conv1=SingleBox(in_channels, out_channels, conv, batch, relu,
                      kernel_size, strides, padding, bias, leaky_relu)
    def forward(self, x):
        return self.relu(self.leaky_relu,torch.cat((self.conv0(x),self.conv1(x)),dim=1))
class DoubleBoxParallelResnet(ResidualBox):
    def __init__(self, in_channels, out_channels, conv, batch, relu, kernel_size, strides, padding, bias=False, leaky_relu=0.1):
        super().__init__(in_channels, out_channels, conv, batch,
                         relu, kernel_size, strides, padding, bias, leaky_relu)
        # cat the output of the two convolutions
        self.core = DoubleBoxParallel(in_channels, out_channels, conv, batch, relu,
                      kernel_size, strides, padding, bias, leaky_relu)
    
    
class oldDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, strides, padding,bias=False,dimensions=2):
        super(oldDoubleConv, self).__init__()
        if dimensions==2:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding, bias=bias),
                nn.BatchNorm2d(num_features=out_channels),
                nn.LeakyReLU(0.1,inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size, strides, padding, bias=bias),
                nn.BatchNorm2d(num_features=out_channels),
                nn.LeakyReLU(0.1,inplace=True)
            )
        if dimensions==3:
            self.conv = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, strides, padding, bias=bias),
                nn.BatchNorm3d(num_features=out_channels),
                nn.LeakyReLU(0.1,inplace=True),
                nn.Conv3d(out_channels, out_channels, kernel_size, strides, padding, bias=bias),
                nn.BatchNorm3d(num_features=out_channels),
                nn.LeakyReLU(0.1,inplace=True)
            )

    def forward(self, x):
        return self.conv(x)
class oldEMUNetv2(nn.Module):
       def __init__(self, in_channels=1, out_channels=1, init_features=32,dimensions=3,number_of_levels=4,batchnorm=False,):
        super().__init__()
        if dimensions == 2:
            self.C = nn.Conv2d
            self.B = nn.BatchNorm2d
            self.P = nn.MaxPool2d
            self.T = nn.ConvTranspose2d
        elif dimensions == 3:
            self.C = nn.Conv3d
            self.B = nn.BatchNorm3d
            self.P = nn.MaxPool3d
            self.T = nn.ConvTranspose3d

        features = init_features
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.ups = nn.ModuleList()
        self.bottleneck = oldDoubleConv(features * 8, features * 16)
        self.output = nn.Conv2d(features, out_channels, kernel_size=1)


class EMUNet(nn.Module):
    def __init__(self, in_channels, out_channels=1, features=[64, 128, 256, 512], dimensions=2, kernel_size=3, padding_size=1, stride_size=1, leaky_relu=0.1, boxtype=SingleBox, batchinbox=False, bias=False):
        self.R = nn.LeakyReLU
        if dimensions == 2:
            self.C = nn.Conv2d
            self.B = nn.BatchNorm2d
            self.P = nn.MaxPool2d
            self.T = nn.ConvTranspose2d
        elif dimensions == 3:
            self.C = nn.Conv3d
            self.B = nn.BatchNorm3d
            self.P = nn.MaxPool3d
            self.T = nn.ConvTranspose3d
        
        self.bias=bias

        if not batchinbox:
            self.B = None

        self.box = boxtype

        super().__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        self.bottleneckNN = self.box(
            in_channels=features[-1],
            out_channels=features[-1]*2,
            conv=self.C,
            batch=self.B,
            relu=self.R,
            kernel_size=kernel_size,  # 3
            strides=stride_size,  # 1
            padding=padding_size,  # 1
            bias=bias,
            leaky_relu=leaky_relu
        )

        self.output = self.C(
                in_channels=features[0],
                out_channels=out_channels,
                kernel_size=1,
                # stride=stride_size,
                # padding=padding_size,
                bias=bias
            )

        self.pool = self.P(kernel_size=2, stride=2, padding=0)
        in_channels_iter = in_channels

        for feature in features:
            self.downs.append(self.box(
                in_channels=in_channels_iter,
                out_channels=feature,
                conv=self.C,
                batch=self.B,
                relu=self.R,
                kernel_size=kernel_size,  # 3
                strides=stride_size,  # 1
                padding=padding_size,  # 1
                bias=bias,
                leaky_relu=leaky_relu

            ))
            in_channels_iter = feature

        for feature in reversed(features):
            up = nn.Sequential(
                self.T(
                    in_channels=feature*2,
                    out_channels=feature,
                    kernel_size=2,
                    stride=2,
                    padding=0
                ),
                self.box(
                    in_channels=feature*2,
                    out_channels=feature,
                    conv=self.C,
                    batch=self.B,
                    relu=self.R,
                    kernel_size=kernel_size,  # 3
                    strides=stride_size,  # 1
                    padding=padding_size,  # 1
                    bias=bias,
                    leaky_relu=leaky_relu
                )
            )

            self.ups.append(up)
    def bottleneck(self,x,params=None):
        return self.bottleneckNN(x)
    
    def forward(self, x,params=None):
        skip_connections = []
        #calculate first order statistics from x
        
        
        for it, down in enumerate(self.downs):
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        x = self.bottleneck(x,params=params)
        skip_connections = skip_connections[::-1]
        for i in range(len(self.ups)):
            x = self.ups[i][0](x)  # Pass through ConvTranspose first
            skip_connection = skip_connections[i]
            # If the height and width of output tensor and skip connection
            # is not same then resize the tensor
            if x.shape != skip_connection.shape:
                U = torch.nn.Upsample(size=skip_connection.shape[2:],)
                x = U(x)
            # Concat the output tensor with skip connection
            concat_x = torch.cat((skip_connection, x), dim=1)

            # Pass the concatinated tensor through DoubleCOnv
            x = self.ups[i][1](concat_x)

        return self.output(x)
    
    
class EMUnetPlus(EMUNet):
    """A subclass of EMUNet that extends its functionality by having the possibility of add some parameters in the bottleneck.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        image_size (tuple): Size of the input image.
        nscalars (int): Number of scalar parameters.
        features (list, optional): List of feature maps for each layer. Defaults to [64, 128, 256, 512].
        dimensions (int, optional): Number of dimensions for the input image. Defaults to 2.
        kernel_size (int, optional): Size of the convolutional kernel. Defaults to 3.
        padding_size (int, optional): Size of the padding. Defaults to 1.
        stride_size (int, optional): Size of the stride. Defaults to 1.
        leaky_relu (float, optional): Negative slope for LeakyReLU activation. Defaults to 0.1.
        boxtype (type, optional): Type of bounding box. Defaults to SingleBox.
        batchinbox (bool, optional): Whether to use batch normalization in the bounding box. Defaults to False.
        bias (bool, optional): Whether to include bias in the linear layer. Defaults to False.
    """
    def __init__(self, in_channels, out_channels, image_size, nscalars, features=[64, 128, 256, 512], dimensions=2, kernel_size=3, padding_size=1, stride_size=1, leaky_relu=0.1, boxtype=SingleBox, batchinbox=False, bias=False):
        super().__init__(in_channels=in_channels, out_channels=out_channels, features=features, dimensions=dimensions, kernel_size=kernel_size, padding_size=padding_size, stride_size=stride_size, leaky_relu=leaky_relu, boxtype=boxtype, batchinbox=batchinbox, bias=bias)
        self.NDOWN = len(self.downs) * 2
        self.image_size_d = np.prod(np.array([a // self.NDOWN for a in image_size]))
        initial_parameters = self.image_size_d * features[-1]
        self.fc = nn.Linear(initial_parameters + nscalars, initial_parameters, bias=self.bias)
        
    def bottleneck(self, x, parameters):
        """Perform the bottleneck operation on the input.

        Args:
            x (torch.Tensor): The input variable at the bottleneck.
            parameters (torch.Tensor): Additional parameters to add at the bottleneck.

        Returns:
            torch.Tensor: The output of the bottleneck operation.
        """        
        O = x.shape
        x = self.fc(torch.cat((x.view(-1), parameters)))
        x = x.view(O)
        # the params are already embedded in the x parameter
        return super().bottleneck(x,params=None)

class EMUnetSegmentation(EMUNet):
    def forward(self, x,params=None):
        S=nn.Softmax(dim=1)
        x = super().forward(x,params=params)
        return S(x)


class EMUnetSegmentationPlus(EMUnetPlus):
    def forward(self, x,params=None):
        S=nn.Softmax(dim=1)
        x = super().forward(x,params=params)
        return S(x)

try:
    import RUNetfeatures
except:
    import treno.RUNetfeatures as RUNetfeatures
class EMRUnet(EMUnetPlus):
    def __init__(self, in_channels, out_channels, image_size, nscalars, features=[64, 128, 256, 512], dimensions=2, kernel_size=3, padding_size=1, stride_size=1, leaky_relu=0.1, boxtype=SingleBox, batchinbox=False,bias=False,version=1):
        super().__init__(in_channels=in_channels, out_channels=out_channels, features=features, dimensions=dimensions, kernel_size=kernel_size, padding_size=padding_size, stride_size=stride_size, leaky_relu=leaky_relu, boxtype=boxtype, batchinbox=batchinbox, bias=bias,image_size=image_size)
        numberoffeatures=0
        self.version=version
        if version==1:
            numberoffeatures=20*in_channels
        else:
            raise ValueError("Version not implemented")
        initial_parameters=self.image_size_d*features[-1]
        self.fc = nn.Linear(initial_parameters+ nscalars+numberoffeatures,initial_parameters ,bias=self.bias)
    def forward(self, x,params=None):
        # S=nn.Softmax(dim=1)
        o=RUNetfeatures.calculate_statistics(x,version=self.version)
        if params is not None:
            params=torch.cat((params,o))
        else:
            params=o
        x = super().forward(x,params=params)
        return x

class EMRUnetSegmentation(EMRUnet):
    def forward(self, x,params=None):
        S=nn.Softmax(dim=1)
        x = super().forward(x,params=params)
        return S(x)
class EMRUnetSegmentationDev(EMRUnet):
    def __init__(self, in_channels, out_channels, image_size, nscalars, features=[64, 128, 256, 512], dimensions=2, kernel_size=3, padding_size=1, stride_size=1, leaky_relu=0.1, boxtype=SingleBox, batchinbox=False,bias=False,version=1,roifeatures=False):
        super().__init__(in_channels=in_channels, out_channels=out_channels, features=features, dimensions=dimensions, kernel_size=kernel_size, padding_size=padding_size, stride_size=stride_size, leaky_relu=leaky_relu, boxtype=boxtype, batchinbox=batchinbox, bias=bias,image_size=image_size)
        numberoffeatures=0
        self.version=version
        # self.roifeatures=roifeatures
        # self.fc2=None
        initial_parameters=self.image_size_d*features[-1]
        
        if version==1:
            numberoffeatures=20*in_channels
            # if roifeatures:
            #     self.fc2 = nn.Linear(initial_parameters + nscalars, initial_parameters, bias=self.bias)
        else:
            raise ValueError("Version not implemented")
        
        self.fc = nn.Linear(initial_parameters+ nscalars+numberoffeatures,initial_parameters ,bias=self.bias)
    def forward(self, x,params=None):
        S=nn.Softmax(dim=1)
        o=RUNetfeatures.calculate_statistics(x,version=self.version)
        if params is not None:
            params=torch.cat((params,o))
        else:
            params=o
        x = super().forward(x,params=params)
        
        return x
    
class EMLeNetPlus(EMUnetPlus):
    def __init__(self, in_channels, out_channels, image_size, nscalars=0, features=[64, 128, 256, 512], dimensions=2, kernel_size=3, padding_size=1, stride_size=1, leaky_relu=0.1, resnet=False, parallelbox=False,batchinbox=False, bias=False,fullyconnectedfeatures=[1024,512]):
        super().__init__(in_channels=in_channels, out_channels=out_channels, features=features, dimensions=dimensions, kernel_size=kernel_size, padding_size=padding_size, stride_size=stride_size, leaky_relu=leaky_relu, boxtype=boxtype, batchinbox=batchinbox, bias=bias,image_size=image_size)
        initial_parameters=int(self.image_size_d*features[0])
        self.fucon=[]
        self.fuconre=[]
        self.fucon.append(nn.Linear(in_features=initial_parameters, out_features=fullyconnectedfeatures[0]))
        self.fuconre.append(nn.LeakyReLU(leaky_relu))
        for n,a in enumerate(fullyconnectedfeatures[1:]):
            self.fucon.append(nn.Linear(in_features=fullyconnectedfeatures[n-1], out_features=a))
            self.fuconre.append(nn.LeakyReLU(leaky_relu))
    def forward(self, x, params=None):
        for down in self.downs:
            x = down(x)
            x = self.pool(x)
        x=self.bottleneck(x,parameters=params)
        for f,r in zip(self.fucon,self.fuconre):
            x = f(x)
            x = r(x)
        return x



class CNNLeNET(nn.Module):
    #  Determine what layers and their order in CNN object
    def __init__(self, dimension, imageSize, inchan=2, outchan=1, featuresmaps=[16, 32, 64]):
        super().__init__()
        self.featuresmaps = featuresmaps
        # initialize first set of CONV => RELU => POOL layers
        if dimension == 2:
            C = nn.Conv2d
            P = nn.MaxPool2d
        else:
            C = nn.Conv3d
            P = nn.MaxPool3d

        self.conv1 = C(in_channels=inchan,
                       out_channels=featuresmaps[0], kernel_size=3, padding=1)
        self.relu1 = nn.LeakyReLU(0.1)
        self.maxpool1 = P(kernel_size=2, stride=2)
        # initialize second set of CONV => RELU => POOL layers
        self.conv2 = C(
            in_channels=featuresmaps[0], out_channels=featuresmaps[1], kernel_size=3, padding=1)
        self.relu2 = nn.LeakyReLU(0.1)
        self.maxpool2 = P(kernel_size=2, stride=2)

        # initialize first (and only) set of FC => RELU layers
        tmp = int(np.prod([x/4 for x in imageSize])*featuresmaps[1])

        self.fc1 = nn.Linear(in_features=tmp, out_features=featuresmaps[2])
        self.relu3 = nn.LeakyReLU(0.1)
        # initialize our softmax classifier
        self.fc2 = nn.Linear(in_features=featuresmaps[2], out_features=outchan)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.maxpool1(out)

        out = self.conv2(out)
        out = self.relu2(out)
        out = self.maxpool2(out)

        # flatten the output from the previous layer and pass it
        # through our only set of FC => RELU layers

        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.relu3(out)
        # pass the output to our softmax classifier to get our output
        # predictions
        out = self.fc2(out)
        out = nn.LeakyReLU(out)
        return out


class CNNLeNETClassifier(CNNLeNET):

    def forward(self, x):
        return torch.sigmoid(self.R(super().forward(x)))

from torchviz import make_dot
def drawmodel(out,model,fn):
    make_dot(out, params=dict(model.named_parameters()), show_saved=True,show_attrs=True).render(fn, format="png")



if __name__ == "__main__":
    NC=2
    IMS=11
    image = torch.randn((1, NC, IMS,IMS))
    model = EMUNet(in_channels=NC,features=[2,4],out_channels=1, dimensions=2, kernel_size=3,resnet=False,image_size=(IMS,IMS),parallelbox=True)
    model2 = EMRUnetSegmentation(in_channels=NC,features=[2,4],out_channels=1, dimensions=2, kernel_size=3,resnet=True,nscalars=2,image_size=(IMS,IMS),parallelbox=True)
    out = model(image)
    out2 = model2(image,torch.from_numpy(np.array([1,2])))
    
    print(out.shape)
    print(out2.shape)
    

   
