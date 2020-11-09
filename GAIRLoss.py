from models import *
import torch
import numpy as np

def GAIRLoss(logit, target, num_steps, pgd_steps, beta, loss_fn, category, nat_logit=None, adv_or_nat=None ,adv_boost=None, nat_boost=None):
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
    elif category == "TRADES":
        batch_size = len(target)
        criterion_kl = nn.KLDivLoss(reduce=False).cuda()
        if adv_or_nat == "adv":
            if adv_boost == False:
                loss = (1.0 / batch_size) * criterion_kl(F.log_softmax(logit, dim=1),
                                                         F.softmax(nat_logit, dim=1)).sum()
            else:
                if loss_fn == 'Tanh':
                    reweight = ((beta+(int(num_steps/2)-pgd_steps)*5/(int(num_steps/2))).tanh()+1)/2
                    normalized_reweight = reweight * len(reweight) / reweight.sum()
                    loss = (1.0 / batch_size) * criterion_kl(F.log_softmax(logit, dim=1),
                                                         F.softmax(nat_logit, dim=1)).sum(dim=1).mul(normalized_reweight).sum()
                elif loss_fn == 'Sigmoid':
                    reweight = (beta+(int(num_steps/2)-pgd_steps)*5/(int(num_steps/2))).sigmoid()
                    normalized_reweight = reweight * len(reweight) / reweight.sum()
                    loss = (1.0 / batch_size) * criterion_kl(F.log_softmax(logit, dim=1),
                                                         F.softmax(nat_logit, dim=1)).sum(dim=1).mul(normalized_reweight).sum()
                elif loss_fn == 'Discrete':
                    reweight = ((num_steps+1)-pgd_steps)/(num_steps+1)
                    normalized_reweight = reweight * len(reweight) / reweight.sum()
                    loss = (1.0 / batch_size) * criterion_kl(F.log_softmax(logit, dim=1),
                                                         F.softmax(nat_logit, dim=1)).sum(dim=1).mul(normalized_reweight).sum()
        elif adv_or_nat == "nat":
            if nat_boost == False:
                loss = nn.CrossEntropyLoss(reduction='mean')(nat_logit, target)
            else:
                if loss_fn == 'Tanh':
                    reweight = ((beta+(int(num_steps/2)-pgd_steps)*5/(int(num_steps/2))).tanh()+1)/2
                    normalized_reweight = reweight * len(reweight) / reweight.sum()
                    loss = nn.CrossEntropyLoss(reduce=False)(nat_logit, target).mul(normalized_reweight).mean()
                elif loss_fn == 'Sigmoid':
                    reweight = (beta+(int(num_steps/2)-pgd_steps)*5/(int(num_steps/2))).sigmoid()
                    normalized_reweight = reweight * len(reweight) / reweight.sum()
                    loss = nn.CrossEntropyLoss(reduce=False)(nat_logit, target).mul(normalized_reweight).mean()
                elif loss_fn == 'Discrete':
                    reweight = ((num_steps+1)-pgd_steps)/(num_steps+1)
                    normalized_reweight = reweight * len(reweight) / reweight.sum()
                    loss = nn.CrossEntropyLoss(reduce=False)(nat_logit, target).mul(normalized_reweight).mean()
            
    return loss
