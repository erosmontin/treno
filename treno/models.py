# unet.py file

import numpy as np
from torch import nn
import torch
import torchvision.transforms.functional as TF
# https://pyimagesearch.com/2021/11/08/u-net-training-image-segmentation-models-in-pytorch/
# https://amaarora.github.io/2020/09/13/unet.html
# https://idiomaticprogrammers.com/post/unet-architecture/


class SingleBox(nn.Module):
    def __init__(self, in_channels, out_channels, conv, batch, relu, kernel_size, strides, padding, bias=False, leaky_relu=0.1):
        super().__init__()
        if batch is not None:
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
    def __init__(self, in_channels, out_channels, conv, batch, relu, kernel_size, strides, padding, bias=False, leaky_relu=0.1):
        super().__init__(in_channels, out_channels, conv, batch,
                         relu, kernel_size, strides, padding, bias, leaky_relu)
        self.conv = nn.Sequential(
            SingleBox(in_channels, out_channels, conv, batch, relu,
                      kernel_size, strides, padding, bias, leaky_relu),
            SingleBox(in_channels, out_channels, conv, batch, relu,
                      kernel_size, strides, padding, bias, leaky_relu),
        )


class EMUNet(nn.Module):
    def __init__(self, in_channels, out_channels=1, features=[64, 128, 256, 512], dimensions=2, kernel_size=3, padding_size=1, stride_size=1, leaky_relu=0.1, resnet=False, batchinbox=False, bias=False):
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

        if not batchinbox:
            self.B = None

        self.box = SingleBox
        if resnet:
            self.box = DoubleBox

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
                kernel_size=kernel_size,
                stride=stride_size,
                padding=padding_size,
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
    def bottleneck(self,x):
        return self.bottleneckNN(x)
    
    def forward(self, x,params=None):
        skip_connections = []
        for it, down in enumerate(self.downs):
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        x = self.bottleneck(x,params)
        print(x.shape)
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
    def __init__(self, in_channels, out_channels,image_size, nscalars,features=[64, 128, 256, 512], dimensions=2, kernel_size=3, padding_size=1, stride_size=1, leaky_relu=0.1, resnet=False, batchinbox=False, bias=False,):
        super().__init__(in_channels, out_channels, features, dimensions, kernel_size, padding_size, stride_size, leaky_relu, resnet, batchinbox, bias)
        self.NDOWN=len(self.downs)*2
        self.image_size_d=np.prod(np.array([a//self.NDOWN for a in image_size]))
        initial_parameters=self.image_size_d*features[-1]
        self.fc = nn.Linear(initial_parameters+ nscalars,initial_parameters ,bias=True)
        
    def bottleneck(self, x,parameters):
        """_summary_

        Args:
            x (_type_): the input variable at the bottleneck
            parameters (_type_): n parwmters to add at the bottleneck

        Returns:
            _type_: the classical bottleneck of a cnn
        """        
        O=x.shape
        x = self.fc(torch.concat((x.view(-1),parameters)))
        x=x.view(O)
        return super().bottleneck(x)

class EMUnetSegmentation(EMUNet):
    def forward(self, x):
        S=nn.Sigmoid()
        x = super().forward(x)
        return S(x)


class EMUnetSegmentationPlus(EMUnetPlus):
    def forward(self, x,params=None):
        S=nn.Sigmoid()
        x = super().forward(x,params=params)
        return S(x)


class EMLeNetPlus(EMUnetPlus):
    def __init__(self, in_channels, out_channels, image_size, nscalars=0, features=[64, 128, 256, 512], dimensions=2, kernel_size=3, padding_size=1, stride_size=1, leaky_relu=0.1, resnet=False, batchinbox=False, bias=False,fullyconnectedfeatures=[1024,512]):
        super().__init__(in_channels, out_channels, image_size, nscalars, features, dimensions, kernel_size, padding_size, stride_size, leaky_relu, resnet, batchinbox, bias)
        initial_parameters=int(self.image_size_d*features[-1])
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
    NC=1
    IMS=11
    image = torch.randn((1, NC, IMS,IMS))
    model = EMUnetPlus(in_channels=NC,features=[2,4],out_channels=1, dimensions=2, kernel_size=3,resnet=True,nscalars=2,image_size=(IMS,IMS))
    out = model(image,torch.from_numpy(np.array([1,2])))
    print(out.shape)
    

   