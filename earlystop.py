from models import *
import torch
import numpy as np

# Geometry-aware early stopped PGD
def GA_earlystop(model, data, target, step_size, epsilon, perturb_steps, tau, type, random, omega):
    # Based on code from https://github.com/zjfheart/Friendly-Adversarial-Training
    
    model.eval()
    K = perturb_steps
    count = 0

    output_target = []
    output_adv = []
    output_natural = []
    output_Kappa = []

    control = torch.zeros(len(target)).cuda()
    control += tau
    Kappa = torch.zeros(len(data)).cuda()

    if random == False:
        iter_adv = data.cuda().detach()
    else:

        if type == "fat_for_trades" :
            iter_adv = data.detach() + 0.001 * torch.randn(data.shape).cuda().detach()
            iter_adv = torch.clamp(iter_adv, 0.0, 1.0)
        if type == "fat" or "fat_for_mart":
            iter_adv = data.detach() + torch.from_numpy(np.random.uniform(-epsilon, epsilon, data.shape)).float().cuda()
            iter_adv = torch.clamp(iter_adv, 0.0, 1.0)
    iter_clean_data = data.cuda().detach()
    iter_target = target.cuda().detach()
    output_iter_clean_data = model(data)

    while K>0:
        iter_adv.requires_grad_()
        output = model(iter_adv)
        pred = output.max(1, keepdim=True)[1]
        output_index = []
        iter_index = []

        for idx in range(len(pred)):
            if pred[idx] != target[idx]:
                if control[idx]==0:
                    output_index.append(idx)
                else:
                    control[idx]-=1
                    iter_index.append(idx)
            else:
                # Update Kappa
                Kappa[idx] += 1
                iter_index.append(idx)

        if (len(output_index)!=0):
            if (len(output_target) == 0):
                # incorrect adv data should not keep iterated
                output_adv = iter_adv[output_index].reshape(-1, 3, 32, 32).cuda()
                output_natural = iter_clean_data[output_index].reshape(-1, 3, 32, 32).cuda()
                output_target = iter_target[output_index].reshape(-1).cuda()
                output_Kappa = Kappa[output_index].reshape(-1).cuda()
            else:
                # incorrect adv data should not keep iterated
                output_adv = torch.cat((output_adv, iter_adv[output_index].reshape(-1, 3, 32, 32).cuda()), dim=0)
                output_natural = torch.cat((output_natural, iter_clean_data[output_index].reshape(-1, 3, 32, 32).cuda()), dim=0)
                output_target = torch.cat((output_target, iter_target[output_index].reshape(-1).cuda()), dim=0)
                output_Kappa = torch.cat((output_Kappa, Kappa[output_index].reshape(-1).cuda()), dim=0)

        model.zero_grad()
        with torch.enable_grad():
            if type == "fat" or type == "fat_for_mart":
                loss_adv = nn.CrossEntropyLoss()(output, iter_target)
            if type == "fat_for_trades":
                criterion_kl = nn.KLDivLoss(size_average=False).cuda()
                loss_adv = criterion_kl(F.log_softmax(output, dim=1),F.softmax(output_iter_clean_data, dim=1))
        loss_adv.backward(retain_graph=True)
        grad = iter_adv.grad

        if len(iter_index) != 0:
            Kappa = Kappa[iter_index]
            control = control[iter_index]
            iter_adv = iter_adv[iter_index]
            iter_clean_data = iter_clean_data[iter_index]
            iter_target = iter_target[iter_index]
            output_iter_clean_data = output_iter_clean_data[iter_index]
            grad = grad[iter_index]
            eta = step_size * grad.sign()
            iter_adv = iter_adv.detach() + eta + omega * torch.randn(iter_adv.shape).detach().cuda()
            iter_adv = torch.min(torch.max(iter_adv, iter_clean_data - epsilon), iter_clean_data + epsilon)
            iter_adv = torch.clamp(iter_adv, 0, 1)
            count += len(iter_target)
        else:
            return output_adv, output_target, output_natural, count, output_Kappa
        K = K-1

    if (len(output_target) == 0):
        output_target = iter_target.reshape(-1).squeeze().cuda()
        output_adv = iter_adv.reshape(-1, 3, 32, 32).cuda()
        output_natural = iter_clean_data.reshape(-1, 3, 32, 32).cuda()
        output_Kappa = Kappa.reshape(-1).cuda()
    else:
        output_adv = torch.cat((output_adv, iter_adv.reshape(-1, 3, 32, 32)), dim=0).cuda()
        output_target = torch.cat((output_target, iter_target.reshape(-1)), dim=0).squeeze().cuda()
        output_natural = torch.cat((output_natural, iter_clean_data.reshape(-1, 3, 32, 32).cuda()),dim=0).cuda()
        output_Kappa = torch.cat((output_Kappa, Kappa.reshape(-1)),dim=0).squeeze().cuda()
    
    return output_adv, output_target, output_natural, count, output_Kappa
