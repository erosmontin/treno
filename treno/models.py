# unet.py file

from torch import nn
import torch
import torchvision.transforms.functional as TF
# https://pyimagesearch.com/2021/11/08/u-net-training-image-segmentation-models-in-pytorch/
#https://amaarora.github.io/2020/09/13/unet.html

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, strides, padding,bias=False,dimensions=2):
        super(DoubleConv, self).__init__()
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


class UNet(nn.Module):
    def __init__(self, in_channels, num_segmentations=1, features=[64, 128, 256, 512],dimensions=2):
        super(UNet, self).__init__()
        self.dimensions= dimensions
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.bottleneck = DoubleConv(
            in_channels=features[-1],
            out_channels=features[-1]*2,
            kernel_size=3,
            strides=1,
            padding=1,
            dimensions=self.dimensions
        )
        if self.dimensions==2:
            self.output = nn.Conv2d(
                in_channels=features[0],
                out_channels=num_segmentations,
                kernel_size=1
            )
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        elif self.dimensions==3:
            self.output = nn.Conv3d(
                in_channels=features[0],
                out_channels=num_segmentations,
                kernel_size=1
            )
            self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        in_channels_iter = in_channels
        for feature in features:
            self.downs.append(DoubleConv(
                    in_channels=in_channels_iter,
                    out_channels=feature,
                    kernel_size=3,
                    strides=1,
                    padding=1,
                    dimensions=self.dimensions
                ))
            in_channels_iter = feature

        for feature in reversed(features):
            if self.dimensions==2:
                up = nn.Sequential(
                    nn.ConvTranspose2d(
                        in_channels=feature*2,
                        out_channels=feature,
                        kernel_size=2,
                        stride=2,
                        padding=0
                    ),
                    DoubleConv(
                        in_channels=feature*2,
                        out_channels=feature,
                        kernel_size=3,
                        padding=1,
                        strides=1,
                        dimensions=self.dimensions
                    )
                )
            elif self.dimensions==3:
                up = nn.Sequential(
                    nn.ConvTranspose3d(
                        in_channels=feature*2,
                        out_channels=feature,
                        kernel_size=2,
                        stride=2,
                        padding=0
                    ),
                    DoubleConv(
                        in_channels=feature*2,
                        out_channels=feature,
                        kernel_size=3,
                        padding=1,
                        strides=1,
                        dimensions=self.dimensions
                    )
                )

            self.ups.append(up)

    def forward(self, x):
        skip_connections = []
        for it,down in enumerate(self.downs):
        # for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
            

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for i in range(len(self.ups)):
            x = self.ups[i][0](x) # Pass through ConvTranspose first

            skip_connection = skip_connections[i]

            # If the height and width of output tensor and skip connection
            # is not same then resize the tensor
            if x.shape != skip_connection.shape:
                if self.dimensions==2:
                    x = TF.resize(x, size=skip_connection.shape[2:])
                elif self.dimensions==3:
                    U = torch.nn.Upsample(size=skip_connection.shape[2:])
                    x= U(x)


            # Concat the output tensor with skip connection
            concat_x = torch.cat((skip_connection, x), dim=1)

            # Pass the concatinated tensor through DoubleCOnv
            x = self.ups[i][1](concat_x)

        return self.output(x)
class Unetplus(UNet):
    def forward(self, x,info_matrix=None):
        skip_connections = []
        for it,down in enumerate(self.downs):
        # for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
            

        x = self.bottleneck(x)
        if not info_matrix is None:
            S=x.shape
            I=info_matrix.shape
            oldNF=S[1]
            newNF=oldNF+I[1]
            C=DoubleConv(
                    in_channels=newNF,
                    out_channels=oldNF,
                    kernel_size=3,
                    padding=1,
                    strides=1,
                    dimensions=self.dimensions,
                    bias=True
                )
            info=torch.zeros((S[0],newNF,*S[2:]))
            info[:,0:oldNF]=x
            for s in range(S[0]): # for all batches
                for nv,v in enumerate(info_matrix[s]):
                    info[s,oldNF+nv]=torch.tensor([v]).repeat(S[2:])
            x=C(info)    
        skip_connections = skip_connections[::-1]

        for i in range(len(self.ups)):
            x = self.ups[i][0](x) # Pass through ConvTranspose first

            skip_connection = skip_connections[i]

            # If the height and width of output tensor and skip connection
            # is not same then resize the tensor
            if x.shape != skip_connection.shape:
                if self.dimensions==2:
                    x = TF.resize(x, size=skip_connection.shape[2:])
                elif self.dimensions==3:
                    U = torch.nn.Upsample(size=skip_connection.shape[2:])
                    x= U(x)

            # Concat the output tensor with skip connection
            concat_x = torch.cat((skip_connection, x), dim=1)

            # Pass the concatinated tensor through DoubleCOnv
            x = self.ups[i][1](concat_x)
        return self.output(x)

if __name__ == "__main__":
    image = torch.randn((2, 3, 50, 50,50))
    model = Unetplus(in_channels=3,dimensions=3)
    out = model(image,torch.randn(2,4))
    print(image.shape, out.shape)
    assert out.shape == (2, 1, 50, 50,50)
