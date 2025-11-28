import torch
import torch.nn as nn

from torch.utils.data import Dataset

import pandas as pd
import numpy as np
import json
from pathlib import Path

# Use new pyable-dataloader
try:
    from pyable_dataloader import PyableDataset, Compose, IntensityNormalization, RandomFlip
    PYABLE_DATALOADER_AVAILABLE = True
except ImportError:
    PYABLE_DATALOADER_AVAILABLE = False
    print("Warning: pyable-dataloader not available. Install with: pip install -e /path/to/pyable-dataloader")

# Use modern pyable package
try:
    from pyable.imaginable import SITKImaginable, Roiable, LabelMapable
    PYABLE_AVAILABLE = True
except ImportError:
    try:
        # Fallback to old package name
        from pyable_eros_montin import imaginable as ima
        PYABLE_AVAILABLE = True
        # Create aliases for compatibility
        SITKImaginable = ima.Imaginable
        Roiable = ima.Roiable
        LabelMapable = ima.LabelMapable
    except ImportError:
        PYABLE_AVAILABLE = False
        print("Warning: pyable not available")

try:
    from pynico_eros_montin import pynico as pn
except ImportError:
    pn = None
# ============================================================================
# LEGACY FUNCTIONS (kept for backward compatibility)
# ============================================================================

def normalize(xI,transform,other=None):
    """Legacy normalization function. Consider using IntensityNormalization from pyable-dataloader."""
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
        


# ============================================================================
# MODERN DATASET CLASSES (using pyable-dataloader)
# ============================================================================

class TrenoDataset(Dataset):
    """
    Modern dataset class using pyable-dataloader.
    
    This replaces the legacy ImageImageDataset, ImageLabelmapDataset classes
    with a unified interface based on pyable-dataloader's PyableDataset.
    
    Args:
        manifest: Path to JSON manifest or CSV file, or dict manifest
        target_size: Target image size [D, H, W] or [H, W]
        target_spacing: Target spacing in mm (float or list)
        transforms: Compose object with transforms
        roi_mask: Whether to apply ROI masking
        roi_dilation: Dilation radius for ROI in mm
        cache_dir: Directory for caching preprocessed data
        return_meta: Whether to return metadata
        stack_channels: Whether to stack multiple images as channels
        
    Example:
        >>> from treno.loaders import TrenoDataset, create_manifest_from_csv
        >>> from pyable_dataloader import Compose, IntensityNormalization, RandomFlip
        >>> 
        >>> # Convert old CSV format to manifest
        >>> manifest = create_manifest_from_csv('train.csv')
        >>> 
        >>> # Create transforms
        >>> transforms = Compose([
        ...     IntensityNormalization(method='zscore'),
        ...     RandomFlip(axes=[1, 2], prob=0.5)
        ... ])
        >>> 
        >>> # Create dataset
        >>> dataset = TrenoDataset(
        ...     manifest=manifest,
        ...     target_size=[64, 64, 64],
        ...     target_spacing=2.0,
        ...     transforms=transforms,
        ...     cache_dir='./cache'
        ... )
        >>> 
        >>> # Use with DataLoader
        >>> from torch.utils.data import DataLoader
        >>> loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)
    """
    
    def __init__(
        self,
        manifest,
        target_size=None,
        target_spacing=None,
        transforms=None,
        roi_mask=False,
        roi_dilation=None,
        cache_dir=None,
        return_meta=False,
        stack_channels=True,
        **kwargs
    ):
        if not PYABLE_DATALOADER_AVAILABLE:
            raise ImportError(
                "pyable-dataloader is required for TrenoDataset. "
                "Install with: pip install -e /path/to/pyable-dataloader"
            )
        
        self.dataset = PyableDataset(
            manifest=manifest,
            target_size=target_size,
            target_spacing=target_spacing,
            transforms=transforms,
            roi_mask=roi_mask,
            roi_dilation=roi_dilation,
            cache_dir=cache_dir,
            return_meta=return_meta,
            stack_channels=stack_channels,
            **kwargs
        )
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        """
        Returns:
            dict with keys:
                - 'images': torch.Tensor [C, D, H, W] or [C, H, W]
                - 'rois': torch.Tensor or None
                - 'labelmaps': torch.Tensor or None
                - 'label': torch.Tensor (scalar)
                - 'id': str
                - 'meta': dict (if return_meta=True)
        """
        return self.dataset[idx]
    
    def get_original_space_overlayer(self, subject_id):
        """Get function to overlay predictions back to original space."""
        return self.dataset.get_original_space_overlayer(subject_id)


def create_manifest_from_csv(
    csv_file,
    image_columns=None,
    roi_column=None,
    labelmap_column=None,
    label_column='label',
    id_column=None
):
    """
    Convert legacy CSV format to manifest dict for TrenoDataset.
    
    Args:
        csv_file: Path to CSV file
        image_columns: List of column names for images, or None to auto-detect
        roi_column: Column name for ROI, or None
        labelmap_column: Column name for labelmap, or None
        label_column: Column name for classification label
        id_column: Column name for subject ID, or None to generate
    
    Returns:
        dict: Manifest compatible with PyableDataset
        
    Example:
        >>> # For CSV with format: label,image1,image2
        >>> manifest = create_manifest_from_csv(
        ...     'train.csv',
        ...     image_columns=['image1', 'image2'],
        ...     label_column='label'
        ... )
        
        >>> # For CSV with format: id,image,roi,label
        >>> manifest = create_manifest_from_csv(
        ...     'train.csv',
        ...     image_columns=['image'],
        ...     roi_column='roi',
        ...     label_column='label',
        ...     id_column='id'
        ... )
    """
    df = pd.read_csv(csv_file)
    
    # Auto-detect image columns if not provided
    if image_columns is None:
        # Assume all columns except label, roi, labelmap, id are images
        exclude = {label_column, roi_column, labelmap_column, id_column}
        image_columns = [col for col in df.columns if col not in exclude and col]
    
    manifest = {}
    
    for idx, row in df.iterrows():
        # Generate or extract subject ID
        if id_column and id_column in df.columns:
            subject_id = str(row[id_column])
        else:
            subject_id = f"subject_{idx:04d}"
        
        # Extract image paths
        image_paths = [str(row[col]) for col in image_columns if col in df.columns and pd.notna(row[col])]
        
        # Build manifest entry
        entry = {"images": image_paths}
        
        # Add ROI if present
        if roi_column and roi_column in df.columns and pd.notna(row[roi_column]):
            entry["rois"] = [str(row[roi_column])]
        
        # Add labelmap if present
        if labelmap_column and labelmap_column in df.columns and pd.notna(row[labelmap_column]):
            entry["labelmaps"] = [str(row[labelmap_column])]
        
        # Add classification label
        if label_column and label_column in df.columns:
            entry["label"] = float(row[label_column])
        
        manifest[subject_id] = entry
    
    return manifest


def save_manifest(manifest, output_path):
    """Save manifest dict to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"✅ Saved manifest to {output_path}")


def load_manifest(manifest_path):
    """Load manifest from JSON file."""
    with open(manifest_path, 'r') as f:
        return json.load(f)


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_treno_dataset_from_csv(
    csv_file,
    target_size=[64, 64, 64],
    target_spacing=2.0,
    augmentation=True,
    cache_dir='./cache',
    **kwargs
):
    """
    Convenience function to create TrenoDataset directly from CSV file.
    
    Args:
        csv_file: Path to CSV file
        target_size: Target image size
        target_spacing: Target spacing in mm
        augmentation: Whether to apply data augmentation
        cache_dir: Cache directory
        **kwargs: Additional arguments passed to create_manifest_from_csv and TrenoDataset
    
    Returns:
        TrenoDataset instance
        
    Example:
        >>> dataset = create_treno_dataset_from_csv(
        ...     'train.csv',
        ...     target_size=[64, 64, 64],
        ...     target_spacing=2.0,
        ...     augmentation=True
        ... )
    """
    if not PYABLE_DATALOADER_AVAILABLE:
        raise ImportError(
            "pyable-dataloader is required. "
            "Install with: pip install -e /path/to/pyable-dataloader"
        )
    
    # Create manifest from CSV
    manifest = create_manifest_from_csv(csv_file, **kwargs)
    
    # Create transforms if augmentation is enabled
    transforms = None
    if augmentation:
        transforms = Compose([
            IntensityNormalization(method='zscore'),
            RandomFlip(axes=[1, 2], prob=0.5)
        ])
    
    # Create dataset
    return TrenoDataset(
        manifest=manifest,
        target_size=target_size,
        target_spacing=target_spacing,
        transforms=transforms,
        cache_dir=cache_dir
    )


# ============================================================================
# MAIN (for testing)
# ============================================================================

if __name__=="__main__":
    print("=" * 60)
    print("Treno Loaders Module")
    print("=" * 60)
    print(f"pyable-dataloader available: {PYABLE_DATALOADER_AVAILABLE}")
    print(f"pyable available: {PYABLE_AVAILABLE}")
    print("=" * 60)
    
    # Example usage of modern dataset
    if PYABLE_DATALOADER_AVAILABLE:
        print("\nModern TrenoDataset example:")
        print(">>> from treno.loaders import create_treno_dataset_from_csv")
        print(">>> dataset = create_treno_dataset_from_csv('train.csv')")
        print(">>> from torch.utils.data import DataLoader")
        print(">>> loader = DataLoader(dataset, batch_size=4, shuffle=True)")
        print(">>> batch = next(iter(loader))")
        print(">>> print(batch['images'].shape)")
    
    # Legacy example
    print("\n" + "=" * 60)
    print("Legacy dataset classes still available for backward compatibility:")
    print("- ImageImageDataset")
    print("- ImageLabelmapDataset")
    print("- ImaImaDataset")
    print("- ImaRoiDataset")
    print("=" * 60)
