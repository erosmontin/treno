import torch
import torch.nn as nn

from scipy.ndimage import morphology
def surfd(gt, test, sampling=1, connectivity=1):
    """
    gt is numpy image
    test is a numpy image

    """
    input_1 = np.atleast_1d(gt.astype(bool))
    input_2 = np.atleast_1d(test.astype(bool))
    conn = morphology.generate_binary_structure(input_1.ndim, connectivity)
    S = input_1 ^ morphology.binary_erosion(input_1, conn)
    Sprime = input_2 ^ morphology.binary_erosion(input_2, conn)
    dta = morphology.distance_transform_edt(~S, sampling)
    dtb = morphology.distance_transform_edt(~Sprime, sampling)
    sds = np.concatenate([np.ravel(dta[Sprime != 0]), np.ravel(dtb[S != 0])])
    return sds
class EMDiceLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.eps=1e-7
    def forward(self, y_true, y_pred):
        y_true = y_true.float()
        y_pred = y_pred.float()
        return 1 - (2 * (y_true * y_pred).sum(dim=0) + self.eps) / (y_true.sum(dim=0) + y_pred.sum(dim=0) + self.eps)

class EMJaccardLoss(EMDiceLoss):
    
    def forward(self, target,output):
        output = torch.sigmoid(output)
        target = target.float()
        intersection = (output * target).sum(dim=1)
        union = output.sum(dim=1) + target.sum(dim=1) - intersection
        return 1 - (2 * intersection) / (union + self.eps)
class EMAccuracyLoss(EMDiceLoss):
    # Calculate the accuracy
    def forward(self, y_true, y_pred):
        correct = (y_pred == y_true).sum(dim=0)
        return correct.float() / y_true.size(0)
    
class EMDistanceLoss(EMDiceLoss):
    def forward(self, y_true, y_pred):
        o=surfd(y_true, y_pred, sampling=1, connectivity=1)
        return o.mean()
    
class EMSegmentaitonMultiLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.losses=[]
    def forward(self, y_true, y_pred):
        loss=0.0
        for l in self.losses:
            loss+=l.forward(y_true,y_pred)
        return loss

class EMSegmentaitonMultiLossCE(EMSegmentaitonMultiLoss):
    def __init__(self, weight=None, size_average=True):
        super().__init__()
        self.weight = weight
        self.size_average = size_average
        self.ce=nn.functional.cross_entropy
    def forward(self, y_true, y_pred):
        # Calculate the cross-entropy loss
        loss=super().forward(y_true, y_pred)
        loss+= self.ce(y_pred, y_true, weight=self.weight, size_average=self.size_average)
        return loss

class EMBinarySegmentaitonMultiLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ce=nn.BCELoss()

import sklearn
import numpy as np
def getCEClassWeights(y,classes=None):
    _y=y.flatten()
    if classes is None:
        classes=np.unique(_y)
    return sklearn.utils.class_weight.compute_class_weight('balanced', classes=classes, y=_y)
