from models import *
import torch
import numpy as np

def GAIRLoss(logit, target, num_steps, pgd_steps, beta, loss_fn, category):
    if category == "AT" or "FAT":
        loss = nn.CrossEntropyLoss(reduce=False)(logit, target)
        if loss_fn == "Tanh":
            reweight = ((beta+(int(num_steps/2)-pgd_steps)*5/(int(num_steps/2))).tanh()+1)/2
            normalized_reweight = reweight * len(reweight) / reweight.sum()
            loss = (loss.mul(normalized_reweight)).mean()
        elif loss_fn == "Sigmoid":
            reweight = (beta+(int(num_steps/2)-pgd_steps)*5/(int(num_steps/2))).sigmoid()
            normalized_reweight = reweight * len(reweight) / reweight.sum()
            loss = (loss.mul(normalized_reweight)).mean() 
        elif loss_fn == "Discrete":
            reweight = ((num_steps+1)-pgd_steps)/(num_steps+1)
            normalized_reweight = reweight * len(reweight) / reweight.sum()
            loss = (loss.mul(normalized_reweight)).mean()
            
    return loss
