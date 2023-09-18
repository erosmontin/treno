import torch
import torch.nn as nn
from scipy.ndimage import morphology
import numpy as np
import torch.nn as nn
from torch.nn import functional as F

EPS=1e-3
def per_class_pixel_accuracy(hist,eps=EPS):
    """Computes the average per-class pixel accuracy.

    The per-class pixel accuracy is a more fine-grained
    version of the overall pixel accuracy. A model could
    score a relatively high overall pixel accuracy by
    correctly predicting the dominant labels or areas
    in the image whilst incorrectly predicting the
    possibly more important/rare labels. Such a model
    will score a low per-class pixel accuracy.

    Args:
        hist: confusion matrix.

    Returns:
        avg_per_class_acc: the average per-class pixel accuracy.
    """
    correct_per_class = torch.diag(hist)
    total_per_class = hist.sum(dim=1)
    per_class_acc = correct_per_class / (total_per_class + EPS)
    avg_per_class_acc = nanmean(per_class_acc)
    return avg_per_class_acc



def jaccard_index(hist,eps=EPS):
    """Computes the Jaccard index, a.k.a the Intersection over Union (IoU).

    Args:
        hist: confusion matrix.

    Returns:
        avg_jacc: the average per-class jaccard index.
    """
    A_inter_B = torch.diag(hist)
    A = hist.sum(dim=1)
    B = hist.sum(dim=0)
    jaccard = A_inter_B / (A + B - A_inter_B + eps)
    avg_jacc = nanmean(jaccard)
    return avg_jacc


def dice_coefficient(hist,eps=EPS):
    """Computes the Sørensen–Dice coefficient, a.k.a the F1 score.

    Args:
        hist: confusion matrix.

    Returns:
        avg_dice: the average per-class dice coefficient.
    """
    A_inter_B = torch.diag(hist)
    A = hist.sum(dim=1)
    B = hist.sum(dim=0)
    dice = (2 * A_inter_B) / (A + B + eps)
    avg_dice = nanmean(dice)
    return avg_dice
def overall_pixel_accuracy(hist,eps=EPS):
    """Computes the total pixel accuracy.

    The overall pixel accuracy provides an intuitive
    approximation for the qualitative perception of the
    label when it is viewed in its overall shape but not
    its details.

    Args:
        hist: confusion matrix.

    Returns:
        overall_acc: the overall pixel accuracy.
    """
    correct = torch.diag(hist).sum()
    total = hist.sum()
    overall_acc = correct / (total + eps)
    return overall_acc

# https://github.com/kevinzakka/pytorch-goodies/blob/master/metrics.py

def _fast_hist(pred,true, num_classes):
    mask = (true >= 0) & (true < num_classes)
    hist = torch.bincount(
        num_classes * true[mask] + pred[mask],
        minlength=num_classes ** 2,
    ).reshape(num_classes, num_classes).float()
    
    return hist

def nanmean(x):
    """Computes the arithmetic mean ignoring any NaNs."""
    return torch.mean(x[x == x])
import torch.nn as nn

class EMLabelMapLoss(nn.Module):
    def __init__(self,num_classes,logit=False,jacard=False,dice=False,overall_accuracy=False,label_accuracy=False,eps=EPS,avg=True,avoid_classes=[],average_loss=True) -> None:
        super().__init__()
        self.hist=None
        self.logit=logit
        
        self.losses=[]
        self.eps=eps
        if dice:
            self.losses.append(dice_coefficient)
        if jacard:
            self.losses.append(jaccard_index)
        if overall_accuracy:
            self.losses.append(overall_pixel_accuracy)
        if label_accuracy:
            self.losses.append(per_class_pixel_accuracy)
        if len(self.losses)==0:
            raise Exception('select at least one metric!')
        self.avoid_classes=avoid_classes
        self.num_classes=num_classes
        self.average_loss=average_loss

        

    def __calc__hist(self,pred,true,num_classes,avoid_classes=[]):
        """Computes various segmentation metrics on 2D feature maps.

        Args:
            pred: a tensor of shape [B, H, W] or [B, 1, H, W].
            true: a tensor of shape [B, H, W] or [B, 1, H, W].
            num_classes: the number of classes to segment. This number
                should be less than the ID of the ignored class.

        Returns:
            set the hist
        """
        device=pred.device
        hist = torch.zeros((num_classes, num_classes)).to(device)
        for t, p in zip(true, pred):
            hist += _fast_hist(t.flatten(), p.flatten(), num_classes)
        
        if len(avoid_classes)>0:
            final_hist=[]
            for a in range(num_classes):
                if not(a in avoid_classes):
                    line=[]
                    for b in range(num_classes):
                        if not(b in avoid_classes):
                            line.append(hist[a][b].item())
                    final_hist.append(line)
            hist=torch.tensor(final_hist)



        self.hist=hist
        return hist
    def forward(self,pred,true):
        if self.logit:
            #in case it is a probabilistic distribution of the labels put the indexes
            pred=torch.argmax(pred,dim=1)
        #update the hist
        self.__calc__hist(pred,true,self.num_classes,self.avoid_classes)
        #start the loss
        loss=0.0
        for m in self.losses:
            loss+=m(self.hist,self.eps)
        if self.average_loss:
            return 1-loss/len(self.losses)
        else:
            return 1-loss
        
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



# class EMJaccardMetric(EMDiceLoss):
    
#     def forward(self, target,output):
#         output = torch.sigmoid(output)
#         target = target.float()
#         intersection = (output * target).sum(dim=1)
#         union = output.sum(dim=1) + target.sum(dim=1) - intersection
#         return 1 - (2 * intersection) / (union + self.eps)
# class EMAccuracyMetric(EMDiceLoss):
#     # Calculate the accuracy
#     def forward(self, y_true, y_pred):
#         correct = (y_pred == y_true).sum(dim=0)
#         return correct.float() / y_true.size(0)
    
# class EMDistanceMetric(EMDiceLoss):
#     def forward(self, y_true, y_pred):
#         o=surfd(y_true, y_pred, sampling=1, connectivity=1)
#         return o.mean()
    
# class EMSegmentaitonMultiMetric(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.losses=[]
#     def forward(self, y_true, y_pred):
#         loss=0.0
#         for l in self.losses:
#             loss+=l.forward(y_true,y_pred)
#         return loss



def dice3D( logits,true,eps=1e-7):
    """Computes the Sørensen–Dice loss.

    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the dice loss so we
    return the negated dice loss.

    Args:
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        true: a tensor of shape [B, 1, H, W].
        eps: added to the denominator for numerical stability.

    Returns:
        dice_loss: the Sørensen–Dice loss.
    """
    device=logits.device

    num_classes=logits.shape[1]
    num_classes
    if num_classes == 1:
        E=torch.eye(num_classes+1).to(device)
    else:
        E=torch.eye(num_classes).to(device)

    if num_classes == 1:
        true_1_hot = E[true.squeeze(1).long()]
        true_1_hot = true_1_hot.permute(0, 4, 1, 2,3).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = E[true.squeeze(1).long()]
        true_1_hot = true_1_hot.permute(0, 4, 1, 2,3).float()
        probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    dice_loss = (2. * intersection / (cardinality + eps)).mean()
    return dice_loss

class dice_loss3D(nn.Module):
    def forward(self, y_pred, y_true):
        return 1 - dice3D(y_pred,y_true)

def jaccard3D(logits, true, eps=1e-7):
    """Computes the Jaccard loss, a.k.a the IoU loss.

    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the jaccard loss so we
    return the negated jaccard loss.

    Args:
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        true: a tensor of shape [B, H, W] or [B, 1, H, W].
        eps: added to the denominator for numerical stability.

    Returns:
        jacc_loss: the Jaccard loss.
    """
    device=logits.device

    num_classes=logits.shape[1]
    num_classes
    if num_classes == 1:
        E=torch.eye(num_classes+1).to(device)
    else:
        E=torch.eye(num_classes).to(device)

    if num_classes == 1:
        true_1_hot = E[true.squeeze(1).long()]
        true_1_hot = true_1_hot.permute(0, 4, 1, 2,3).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = E[true.squeeze(1).long()]
        true_1_hot = true_1_hot.permute(0, 4, 1, 2,3).float()
        probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    union = cardinality - intersection
    jacc_loss = (intersection / (union + eps)).mean()
    return jacc_loss

class jacard_loss3D(nn.Module):
    def forward(self, y_pred, y_true):
        return 1 - jaccard3D(y_pred,y_true)


class EMCrossEntropyLoss(nn.CrossEntropyLoss):
    def forward(self, y_pred, y_true):
        # Calculate the cross-entropy loss squeezing the return
        return super().forward(y_pred, y_true.squeeze(1).long())
    
class EMMulticlassLoss(EMCrossEntropyLoss):
    def __init__(self, weight, size_average=None,dimension=3, ignore_index=-100, reduce=None, reduction='mean', label_smoothing= 0) -> None:
        super().__init__(weight, size_average, ignore_index, reduce, reduction, label_smoothing)
        if dimension==3:
            self.j=jacard_loss3D()
            self.d=dice_loss3D()
    def forward(self,y_pred,y_true):
        l =super().forward(y_pred,y_true)
        d=1-self.d(y_pred,y_true)
        j=1-self.j(y_pred,y_true)
        return (l+d+j)/3.


# def tversky_loss(true, logits, alpha, beta, eps=1e-7):
#     """Computes the Tversky loss [1].

#     Args:
#         true: a tensor of shape [B, H, W] or [B, 1, H, W].
#         logits: a tensor of shape [B, C, H, W]. Corresponds to
#             the raw output or logits of the model.
#         alpha: controls the penalty for false positives.
#         beta: controls the penalty for false negatives.
#         eps: added to the denominator for numerical stability.

#     Returns:
#         tversky_loss: the Tversky loss.

#     Notes:
#         alpha = beta = 0.5 => dice coeff
#         alpha = beta = 1 => tanimoto coeff
#         alpha + beta = 1 => F beta coeff

#     References:
#         [1]: https://arxiv.org/abs/1706.05721
#     """
#     num_classes = logits.shape[1]
#     if num_classes == 1:
#         true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
#         true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
#         true_1_hot_f = true_1_hot[:, 0:1, :, :]
#         true_1_hot_s = true_1_hot[:, 1:2, :, :]
#         true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
#         pos_prob = torch.sigmoid(logits)
#         neg_prob = 1 - pos_prob
#         probas = torch.cat([pos_prob, neg_prob], dim=1)
#     else:
#         true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
#         true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
#         probas = F.softmax(logits, dim=1)
#     true_1_hot = true_1_hot.type(logits.type())
#     dims = (0,) + tuple(range(2, true.ndimension()))
#     intersection = torch.sum(probas * true_1_hot, dims)
#     fps = torch.sum(probas * (1 - true_1_hot), dims)
#     fns = torch.sum((1 - probas) * true_1_hot, dims)
#     num = intersection
#     denom = intersection + (alpha * fps) + (beta * fns)
#     tversky_loss = (num / (denom + eps)).mean()
#     return (1 - tversky_loss)


# class EMSegmentaitonMultiLossCE(EMSegmentaitonMultiLoss):
#     def __init__(self, weight=None, size_average=True):
#         super().__init__()
#         self.weight = weight
#         self.size_average = size_average
#         self.ce=nn.functional.cross_entropy
#     def forward(self, y_true, y_pred):
#         # Calculate the cross-entropy loss
#         loss=super().forward(y_true, y_pred)
#         loss+= self.ce(y_pred, y_true, weight=self.weight, size_average=self.size_average)
#         return loss

# class EMBinarySegmentaitonMultiMetric(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
#         self.ce=nn.BCEMetric()



#special thanks to
# https://github.com/kevinzakka/pytorch-goodies/blob/master/metrics.py