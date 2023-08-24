# import torch

# a=torch.rand((3,3,2))
# print(a)
# O=a.shape
# a=a.view(-1)
# a=a.view(O)
# print(a)

import losses as l
import numpy as np
from pyable_eros_montin import dev as ima
from pynico_eros_montin import pynico as pn
C=pn.Pathable('/data/MYDATA/fulldixon-images/')
p=np.array([])
for d in C.getDirectoriesInPath():
    try:
        F=f'{d}/data/roi.nii.gz'
        L=ima.LabelMapable(F)
        L.changeImageSpacing([30,30,30])
        p=np.concatenate((p,L.getImageAsNumpy().flatten()))
    except:
        print(F)
# [ 0.34685429 18.60086101 15.82679226]
print(l.getCEClassWeights(p))