import torch
def train(model,loss, train_loader,optimizer, epoch,alt_train_loaders=[],writer=None):
    model.train()
    training_loss = 0.0
    for (x, y) in train_loader:
        
        if len(alt_train_loaders):
            for AD in alt_train_loaders:
                (_x, _y)= next(iter(AD))
                x=torch.concat((x,_x),0)
                y=torch.concat((y,_y),0)
        output = model(x)
        l=loss(output, torch.nn.functional.one_hot(y.long(),2).float())
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        training_loss += l.item()
    if writer:
        writer.add_scalar("training_loss", training_loss, epoch)

from sklearn.metrics import confusion_matrix, roc_auc_score, multilabel_confusion_matrix
from pynico_eros_montin import stats as st


import numpy as np

def testPrediction(Ygt, Yhat,labels=None):
    O = []
    if labels is None:
        labels=np.unique(Ygt)
    C=multilabel_confusion_matrix(Ygt.flatten(), Yhat.flatten(),labels=labels)
    for c,l in zip(C,labels):
        tn_, fp_, fn_, tp_ = c.ravel()
        o = {"accuracy": st.accuracyFromConfusion(c),
             "specificity": st.specificityFromConfusion(c),
             "sensitivity": st.sensitivityFromConfusion(c),
             "tn": tn_,
             "tp": tp_,
             "fp": fp_,
             "fn": fn_,
             "label":l
             }
        O.append(o)

    return O