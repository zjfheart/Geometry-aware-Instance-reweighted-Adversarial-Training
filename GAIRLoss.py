from models import *
import torch
import numpy as np

def GAIRLoss(logit, target, num_steps, Kappa, Lambda, loss_fn):
    # Calculate weight assignment according to geometry value
    loss = nn.CrossEntropyLoss(reduce=False)(logit, target)
    if loss_fn == "Tanh":
        reweight = ((Lambda+(int(num_steps/2)-Kappa)*5/(int(num_steps/2))).tanh()+1)/2
        normalized_reweight = reweight * len(reweight) / reweight.sum()
        loss = (loss.mul(normalized_reweight)).mean()
    elif loss_fn == "Sigmoid":
        reweight = (Lambda+(int(num_steps/2)-Kappa)*5/(int(num_steps/2))).sigmoid()
        normalized_reweight = reweight * len(reweight) / reweight.sum()
        loss = (loss.mul(normalized_reweight)).mean() 
    elif loss_fn == "Discrete":
        reweight = ((num_steps+1)-Kappa)/(num_steps+1)
        normalized_reweight = reweight * len(reweight) / reweight.sum()
        loss = (loss.mul(normalized_reweight)).mean()
            
    return loss
