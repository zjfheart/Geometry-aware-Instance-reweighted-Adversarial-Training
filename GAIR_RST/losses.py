"""
Robust training losses. Based on code from
https://github.com/yaodongyu/TRADES
"""

import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

import pdb


def entropy_loss(unlabeled_logits):
    unlabeled_probs = F.softmax(unlabeled_logits, dim=1)
    return -(unlabeled_probs * F.log_softmax(unlabeled_logits, dim=1)).sum(
        dim=1).mean(dim=0)

def cwloss(output, target,confidence=50, num_classes=10):
    # Compute the probability of the label class versus the maximum other
    # The same implementation as in repo CAT https://github.com/sunblaze-ucb/curriculum-adversarial-training-CAT
    target = target.data
    target_onehot = torch.zeros(target.size() + (num_classes,))
    target_onehot = target_onehot.cuda()
    target_onehot.scatter_(1, target.unsqueeze(1), 1.)
    target_var = Variable(target_onehot, requires_grad=False)
    real = (target_var * output).sum(1)
    other = ((1. - target_var) * output - target_var * 10000.).max(1)[0]
    loss = -torch.clamp(real - other + confidence, min=0.)  # equiv to max(..., 0.)
    loss = torch.sum(loss)
    return loss

def pgd_generate(model, data, target, epsilon, step_size, num_steps,loss_fn,category,rand_init):
    model.eval()
    pgd_steps = torch.zeros(len(data))
    if category == "trades":
        x_adv = data.detach() + 0.001 * torch.randn(data.shape).cuda().detach() if rand_init else data.detach()
        nat_output = model(data)
    if category == "Madry":
        x_adv = data.detach() + torch.from_numpy(np.random.uniform(-epsilon, epsilon, data.shape)).float().cuda() if rand_init else data.detach()
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    for k in range(num_steps):
        x_adv.requires_grad_()
        output = model(x_adv)
        #nat_output = model(data)
        predict = output.max(1, keepdim=True)[1]
        for p in range(len(x_adv)):
            if predict[p] == target[p]:
                pgd_steps[p] += 1
        model.zero_grad()
        with torch.enable_grad():
            if loss_fn == "cent":
                loss_adv = nn.CrossEntropyLoss(reduction="mean")(output, target)
            if loss_fn == "cw":
                loss_adv = cwloss(output,target)
            if loss_fn == "kl":
                criterion_kl = nn.KLDivLoss(size_average=False).cuda()
                loss_adv = criterion_kl(F.log_softmax(output, dim=1),F.softmax(nat_output, dim=1))
        loss_adv.backward()
        eta = step_size * x_adv.grad.sign()
        x_adv = x_adv.detach() + eta
        x_adv = torch.min(torch.max(x_adv, data - epsilon), data + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    x_adv = Variable(x_adv, requires_grad=False)
    return x_adv, pgd_steps


def trades_loss(model,
                x_natural,
                y,
                optimizer,
                step_size=0.003,
                epsilon=0.031,
                perturb_steps=10,
                beta=1.0,
                adversarial=True,
                distance='inf',
                entropy_weight=0):
    """The TRADES KL-robustness regularization term proposed by
       Zhang et al., with added support for stability training and entropy
       regularization"""
    if beta == 0:
        logits = model(x_natural)
        loss = F.cross_entropy(logits, y)
        inf = torch.Tensor([np.inf])
        zero = torch.Tensor([0.])
        return loss, loss, inf, zero

    # define KL-loss
    criterion_kl = nn.KLDivLoss(reduction='sum')
    model.eval()  # moving to eval mode to freeze batchnorm stats
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.  # the + 0. is for copying the tensor
    if adversarial:
        if distance == 'l_inf':
            x_adv += 0.001 * torch.randn(x_natural.shape).cuda().detach()

            for _ in range(perturb_steps):
                x_adv.requires_grad_()
                with torch.enable_grad():
                    loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                           F.softmax(model(x_natural), dim=1))
                grad = torch.autograd.grad(loss_kl, [x_adv])[0]
                x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
                x_adv = torch.min(torch.max(x_adv, x_natural - epsilon),
                                  x_natural + epsilon)
                x_adv = torch.clamp(x_adv, 0.0, 1.0)
        else:
            raise ValueError('No support for distance %s in adversarial '
                             'training' % distance)
    else:
        if distance == 'l_2':
            x_adv = x_adv + epsilon * torch.randn_like(x_adv)
        else:
            raise ValueError('No support for distance %s in stability '
                             'training' % distance)

    model.train()  # moving to train mode to update batchnorm stats

    # zero gradient
    optimizer.zero_grad()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    logits_adv = F.log_softmax(model(x_adv), dim=1)
    logits = model(x_natural)

    loss_natural = F.cross_entropy(logits, y, ignore_index=-1)
    p_natural = F.softmax(logits, dim=1)
    loss_robust = criterion_kl(
        logits_adv, p_natural) / batch_size

    loss = loss_natural + beta * loss_robust

    is_unlabeled = (y == -1)
    if torch.sum(is_unlabeled) > 0:
        logits_unlabeled = logits[is_unlabeled]
        loss_entropy_unlabeled = entropy_loss(logits_unlabeled)
        loss = loss + entropy_weight * loss_entropy_unlabeled
    else:
        loss_entropy_unlabeled = torch.tensor(0)

    return loss, loss_natural, loss_robust, loss_entropy_unlabeled

def GAIR_trades_loss(model,
                x_natural,
                y,
                optimizer,
                step_size=0.003,
                epsilon=0.031,
                perturb_steps=10,
                beta=1.0,
                adversarial=True,
                distance='inf',
                entropy_weight=0,
                Lambda=-1.0):
    """The TRADES KL-robustness regularization term proposed by
       Zhang et al., with added support for stability training and entropy
       regularization"""
    Kappa = torch.zeros(len(x_natural))

    if beta == 0:
        logits = model(x_natural)
        loss = F.cross_entropy(logits, y)
        inf = torch.Tensor([np.inf])
        zero = torch.Tensor([0.])
        return loss, loss, inf, zero
    
    # define KL-loss
    criterion_kl = nn.KLDivLoss(reduction="sum")
    model.eval()  # moving to eval mode to freeze batchnorm stats
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.  # the + 0. is for copying the tensor
    if adversarial:
        if distance == 'l_inf':
            x_adv += 0.001 * torch.randn(x_natural.shape).cuda().detach()

            for _ in range(perturb_steps):
                x_adv.requires_grad_()
                mid_logit = model(x_adv)
                mid_predict = mid_logit.max(1, keepdim=True)[1]
                # Update Kappa
                for p in range(len(x_adv)):
                    if mid_predict[p] == y[p]:
                        Kappa[p] += 1
                with torch.enable_grad():
                    loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                           F.softmax(model(x_natural), dim=1))
                grad = torch.autograd.grad(loss_kl, [x_adv])[0]
                x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
                x_adv = torch.min(torch.max(x_adv, x_natural - epsilon),
                                  x_natural + epsilon)
                x_adv = torch.clamp(x_adv, 0.0, 1.0)
        else:
            raise ValueError('No support for distance %s in adversarial '
                             'training' % distance)
    else:
        if distance == 'l_2':
            x_adv = x_adv + epsilon * torch.randn_like(x_adv)
        else:
            raise ValueError('No support for distance %s in stability '
                             'training' % distance)

    if Lambda <= 10.0:
        cw_adv, _ = pgd_generate(model,x_natural, y, epsilon, step_size, perturb_steps, loss_fn="cw",category="trades", rand_init=True)

    
    Kappa = Kappa.cuda()
    model.train()  # moving to train mode to update batchnorm stats

    # zero gradient
    optimizer.zero_grad()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    logits_adv = F.log_softmax(model(x_adv), dim=1)
    if Lambda <= 10.0: 
        logits_cw = F.log_softmax(model(cw_adv),dim=1)
    logits = model(x_natural)

    loss_natural = F.cross_entropy(logits, y, ignore_index=-1, reduction='none')

    loss_natural = loss_natural.mean()
    p_natural = F.softmax(logits, dim=1)

    if Lambda <= 10.0:
        criterion_kl = nn.KLDivLoss(reduce=False).cuda()
        loss_robust = (criterion_kl(logits_adv, p_natural).sum(dim=1).mul(((Lambda+((perturb_steps/2)-Kappa)).tanh()+1)/2)+criterion_kl(logits_cw, p_natural).sum(dim=1).mul(1-((Lambda+((perturb_steps/2)-Kappa)).tanh()+1)/2)).sum()
        loss_robust = loss_robust / batch_size
    else:
        loss_robust = criterion_kl(
            logits_adv, p_natural) / batch_size

    loss = loss_natural + beta * loss_robust

    is_unlabeled = (y == -1)
    if torch.sum(is_unlabeled) > 0:
        logits_unlabeled = logits[is_unlabeled]
        loss_entropy_unlabeled = entropy_loss(logits_unlabeled)
        loss = loss + entropy_weight * loss_entropy_unlabeled
    else:
        loss_entropy_unlabeled = torch.tensor(0)

    return loss, loss_natural, loss_robust, loss_entropy_unlabeled

def noise_loss(model,
               x_natural,
               y,
               epsilon=0.25,
               clamp_x=True):
    """Augmenting the input with random noise as in Cohen et al."""
    # logits_natural = model(x_natural)
    x_noise = x_natural + epsilon * torch.randn_like(x_natural)
    if clamp_x:
        x_noise = x_noise.clamp(0.0, 1.0)
    logits_noise = model(x_noise)
    loss = F.cross_entropy(logits_noise, y, ignore_index=-1)
    return loss

