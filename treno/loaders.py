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
    

def possibletransforms(xI,yI,tr):
    if 'normalizex' in tr.keys():
        xI=normalize(xI,tr["normalizex"])
    if 'normalizey' in tr.keys():
        yI=normalize(yI,tr["normalizey"])
    if 'normalizexv' in tr.keys():
        xI=normalize(xI,"value",tr["normalizexv"])
    if 'normalizeyv' in tr.keys():
        yI=normalize(yI,"value",tr["normalizeyv"])
    if 'resize' in tr.keys():
        xI.changeImageSize(tr["resize"])
        yI.changeImageSize(tr["resize"])
    return xI,yI


class ImageImageDataset(Dataset):
    def __init__(self, annotations_file, transform=None,ausiliary=None):
        self.listofdata = pd.read_csv(annotations_file)
        self.transform = transform
        self.ausiliary=None
        self.first =ima.Imaginable
        self.second =ima.Imaginable
        if ausiliary:
            self.ausiliary=pd.read_csv(ausiliary)

    def __len__(self):
        return len(self.listofdata)
    def __gettheimagesFilename__(self,idx):
        return self.listofdata.iloc[idx, 0],self.listofdata.iloc[idx, 1]
    def __transform__(self,xI,yI):
        if self.transform==None:
            pass
        else:
            xI,yI=possibletransforms(xI,yI,self.transform)
        return xI,yI
    def __theAbles__(self,idx):
        return self.first(filename=self.listofdata.iloc[idx, 0]), self.second(filename=self.listofdata.iloc[idx, 1])
    def __gettheimages__(self,idx):
        
        xI,yI=self.__theAbles__(idx)
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

class ImageLabelmapDataset(ImageImageDataset):
    def __init__(self, annotations_file, transform=None, ausiliary=None,index=None):
        super().__init__(annotations_file, transform, ausiliary)
        self.index=index
        self.second=ima.Roiable
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


class ImaImaDataset(ImageImageDataset):
    def __getitem__(self, idx):
        return ima.Imaginable(filename=self.listofdata.iloc[idx, 0]),ima.Imaginable(filename=self.listofdata.iloc[idx, 1])

class ImaRoiDataset(ImageImageDataset):
    def __getitem__(self, idx):
        return ima.Imaginable(filename=self.listofdata.iloc[idx, 0]),ima.Roiable(filename=self.listofdata.iloc[idx, 1])


import numpy as np

def compute_prob(N,t='fuzzy',other=None):
    N=N.astype(np.float32)
    S=N.shape
    N[N>0]=1
    D=np.zeros(S,dtype=np.float32)

    if t=='padding':
        a=ima.Roiable()
        N[N>0]=1.0
        a.setImageFromNumpy(N)
        a.dilateRadius(int(other))
        D=a.getImageAsNumpy().astype(np.float32)
        
    elif t=='fuzzy':
        a=ima.Imaginable()
        N[N>0]=1.0
        a.setImageFromNumpy(N)
        gaussian = sitk.MeanImageFilter()
        gaussian.SetRadius(int(other))
        D=gaussian.Execute(a.getImage())
        a.setImage(D)
        a.divide(a.getMaximumValue())
        D=a.getImageAsNumpy()
        D/=np.max(D)
        D[N>0]=1.0


    else:
        for t in range(3):
            T=np.expand_dims(N.sum(axis=t),axis=t)
            d=np.ones((1,3),dtype=np.uint64)[0]
            d[t]=S[t]
            D+=np.tile(T,[int(g) for g in d])
        D/=np.max(D)

    return D

def getcenter(D,borders=None,th=0.5):    
    
    MAS=D.shape

    dim=len(MAS)
    MIS=[0]*dim
    B=[0]*dim
    if borders:
        B=borders
    MASF=[np.floor(s-b) for s,b in zip(MAS,B)]
    MISF=[np.ceil(s+b) for s,b in zip(MIS,B)]
    f=0
    while(f<th):
        P=[int(np.random.randint(mi,ma,1)) for mi,ma in zip(MISF,MASF)]
        f=D[tuple(P)]
        for t,b in zip(P,B):
            lmi=t-b
            lma=t+b

    
    return P
def getboundaries(D,size,th=0.5):
    B=[np.ceil(s/2) for s in size]
    C=getcenter(D,B,th)
    return [int(c-b) for c,b in zip(C,B) ],[int(c+b) for c,b in zip(C,B) ]



def cutAndCat(x,y,X,Y,NR,size,th=0.5,sampling='fuzzy',samplingsigma=2.0,penalty=0.1):
    D=compute_prob(y.getImageAsNumpy(),sampling,samplingsigma)
    dim=len(D.shape)
    for a in range(NR):
        
        L,U=getboundaries(D,size,th)
        x.cropImage(L,U)
        y.cropImage(L,U)
        if dim==2:
            D[L[0]:U[0],L[1]:U[1]]-=penalty
        elif dim==3:
            D[L[0]:U[0],L[1]:U[1],L[1]:U[1]]-=penalty
        X[a]=np.expand_dims(x.getImageAsNumpy().astype(np.float32),0)
        Y[a]=np.expand_dims(y.getImageAsNumpy().astype(np.float32),0)
        x.undo()
        y.undo()
    return X,Y

def ImaginableDataloader(x,y,size=[60,60,60],NR=10,transforms={},ND=5,resolution=1,RT=[],th=0.2,sampling='fuzzy',samplingsigma=2.0,penalty=0.1):
    nRT=len(RT) # at least the non rototrnslated
    X=np.zeros([NR+(ND*nRT),1,*size],dtype=np.float32)
    Y=np.zeros([NR+(ND*nRT),1,*size],dtype=np.float32)
    x,y=possibletransforms(x,y,transforms)
    SP=[resolution,resolution,resolution]
    if(x.getImageSpacing()[0]!=resolution):
        x.changeImageSpacing(SP)
        y.changeImageSpacing(SP)
    _in_=0
    _out_=NR
    X[_in_:_out_],Y[_in_:_out_]=cutAndCat(x,y,X[_in_:_out_],Y[_in_:_out_],NR,size,th,sampling=sampling,samplingsigma=samplingsigma,penalty=penalty)
    for ind,t in enumerate(RT):
        x2=x.getDuplicate()
        y2=y.getDuplicate()
        x2.transform(t)
        y2.transform(t)
        in_=NR+(ind)*ND
        out_=NR+(ind+1)*ND
        X[in_:out_],Y[in_:out_]=cutAndCat(x2,y2,X[in_:out_],Y[in_:out_],ND,size,th,sampling=sampling,samplingsigma=samplingsigma,penalty=penalty)


    return torch.from_numpy(X) , torch.from_numpy(Y)
# # https://pytorch.org/docs/stable/data.html
# class myDataLoader(torch.utils.data.DataLoader):
#     def __init__(self) -> None:
#         super().__init__()
#     loader_collate = DataLoader(
#     dataset, shuffle=True, batch_size=5, collate_fn=collate_fn)
        


if __name__=="__main__":
    all_transforms={'resize':[320,320,120],'normalizex':'max'}
    transforms={'normalizex':'max'}
    # train_dataset0 = ImaginableLabelmapDataset('treno/test.txt',transform=all_transforms)
    # # x,y = train_dataset0.__getitem__(0)
    # # print(x.shape)
    # # print(y.shape)
    # test_loader0 = torch.utils.data.DataLoader(dataset = train_dataset0,
    #                                        batch_size = 2,
    #                                        shuffle = False)
    # X,Y=next(iter(test_loader0))
    # print(X.shape)
    # print(Y.shape)
    
    
    train_dataset = ImaRoiDataset('treno/test.txt')
    x,y=train_dataset.__getitem__(1)


    import SimpleITK as sitk
    T=sitk.TranslationTransform(x.getImageDimension(),[5,5,5]),
    S=sitk.ScaleTransform(x.getImageDimension())
    S.SetScale([0.9,0.9,0.9])
    # S.SetCenter()
    X,Y=ImaginableDataloader(x,y,size=[60,60,60],NR=10,transforms=transforms,ND=5,RT=[T,S],th=0.8,sampling='padding',samplingsigma=10)

    print(X)
    print(Y)

    for a in range(X.shape[0]):
        ima.saveNumpy(X[a,0],f'/g/x1{a}.mha')
        ima.saveNumpy(Y[a,0],f'/g/y1{a}.mha')
    # train_dataset1 = ImaginableLabelmapDataset('treno/test.txt',transform=all_transforms,index=[0,1,2])
    # x,y = train_dataset1.__getitem__(0)
    # test_loader1 = torch.utils.data.DataLoader(dataset = train_dataset1,
    #                                        batch_size = 2,
    #                                        shuffle = False)
    # x1,y1=next(iter(test_loader1))
    # print(x1.shape)
    # print(y1.shape)
