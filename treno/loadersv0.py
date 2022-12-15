import torch
import torch.nn as nn


from torch.utils.data import Dataset

import pandas as pd
import numpy as np

from pyable_eros_montin import imaginable as ima
from pynico_eros_montin import pynico as pn
def normalize(xI,transform,other=None):
    xI.cast(float)
    if transform.lower()=='max':
        xI.divide(xI.getMaximumValue())
    elif transform.lower()=='value':
        xI.divide(other)
    elif transform.lower()=='z':
        xI.subtract(xI.getMeanValue())
        xI.divide(xI.getStdValue())
    return xI

def labelMapToChannel(y,include=None):
    if include is None:
        ONC=np.unique(y.flatten())
    else:
        ONC=include
    yo=np.zeros([len(ONC),*y.shape])
    for i,v in enumerate(ONC):
            l=y==v
            yo[i][l]=1
    return yo

class ImaginableDataset(Dataset):
    def __init__(self, annotations_file, transform=None,ausiliary=None):
        self.listofdata = pd.read_csv(annotations_file)
        self.transform = transform
        self.ausiliary=None
        if ausiliary:
            self.ausiliary=pd.read_csv(ausiliary)

    def __len__(self):
        return len(self.listofdata)

    def __transform__(self,xI,yI):
        if self.transform==None:
            pass
        else:
            if 'normalizex' in self.transform.keys():
                xI=normalize(xI,self.transform["normalizex"])
            if 'normalizey' in self.transform.keys():
                yI=normalize(yI,self.transform["normalizex"])
            if 'normalizexv' in self.transform.keys():
                xI=normalize(xI,"value",self.transform["normalizexv"])
            if 'normalizeyv' in self.transform.keys():
                yI=normalize(yI,"value",self.transform["normalizeyv"])
            if 'resize' in self.transform.keys():
                xI.changeImageSize(self.transform["resize"])
                yI.changeImageSize(self.transform["resize"])        
        return xI,yI
    def __gettheimages__(self,idx):
        xI = ima.Imaginable(filename=self.listofdata.iloc[idx, 0])
        
        yI = ima.Imaginable(filename=self.listofdata.iloc[idx, 1])

        xI,yI=self.__transform__(xI,yI)
        # if self.transform:
        x = xI.getImageAsNumpy().astype(np.float32)
        # if self.target_transform:
        y = yI.getImageAsNumpy().astype(np.uint8)
        return x,y

    def __getitem__(self, idx):
        x,y=self.__gettheimages__(idx)
        x=np.expand_dims(x,0)
        y=np.expand_dims(y,0)
        if self.ausiliary:
            aux=np.array(self.ausiliary.iloc[idx].tolist())
            return torch.from_numpy(x) , torch.from_numpy(y), aux
            # return x.type(torch.FloatTensor) , y.type(torch.FloatTensor), aux
        return torch.from_numpy(x) , torch.from_numpy(y)
        # return x.type(torch.FloatTensor) , y.type(torch.FloatTensor)

class ImaginableLabelmapDataset(ImaginableDataset):
    def __init__(self, annotations_file, transform=None, ausiliary=None,index=None):
        super().__init__(annotations_file, transform, ausiliary)
        self.index=index

    def __getitem__(self, idx):
        # output=[NC,*image.size]
        x,y=self.__gettheimages__(idx)
        y=labelMapToChannel(y,self.index)
        x=np.expand_dims(x,0)
        if self.ausiliary:
            aux=np.array(self.ausiliary.iloc[idx].tolist())
            return torch.from_numpy(x) , torch.from_numpy(y), aux
            # return x.type(torch.FloatTensor) , y.type(torch.FloatTensor), aux
        return torch.from_numpy(x) , torch.from_numpy(y)
        # return x.type(torch.FloatTensor) , y.type(torch.FloatTensor)



if __name__=="__main__":
    all_transforms={'resize':[320,320,120],'normalizex':'max'}
    train_dataset0 = ImaginableDataset('treno/test.txt',transform=all_transforms)
    x,y = train_dataset0.__getitem__(0)
    print(x.shape)
    print(y.shape)
    test_loader0 = torch.utils.data.DataLoader(dataset = train_dataset0,
                                           batch_size = 2,
                                           shuffle = False)
    x,y=next(iter(test_loader0))
    print(x.shape)
    print(y.shape)



    # train_dataset1 = ImaginableLabelmapDataset('treno/test.txt',transform=all_transforms,index=[0,1,2])
    # x,y = train_dataset1.__getitem__(0)
    # test_loader1 = torch.utils.data.DataLoader(dataset = train_dataset1,
    #                                        batch_size = 2,
    #                                        shuffle = False)
    # x1,y1=next(iter(test_loader1))
    # print(x1.shape)
    # print(y1.shape)
