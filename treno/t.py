import torch

a=torch.rand((3,3,2))
print(a)
O=a.shape
a=a.view(-1)
a=a.view(O)
print(a)