import os
import time
import argparse
import torchvision
import torch.optim as optim
from torchvision import transforms
import pickle
import datetime
from models import *
from earlystop import earlystop
from GAIRLoss import GAIRLoss
import numpy as np
import attack_generator as attack
from utils import Logger, AverageMeter
from torch.autograd import Variable as V

parser = argparse.ArgumentParser(description='GAIRAT')
parser.add_argument('--epochs', type=int, default=120, metavar='N', help='number of epochs to train')
parser.add_argument('--weight-decay', '--wd', default=2e-4, type=float, metavar='W')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum')
parser.add_argument('--epsilon', type=float, default=0.031, help='perturbation bound')
parser.add_argument('--num-steps', type=int, default=10, help='maximum perturbation step K')
parser.add_argument('--step-size', type=float, default=0.007, help='step size')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed')
parser.add_argument('--net', type=str, default="WRN",help="decide which network to use,choose from smallcnn,resnet18,WRN")
parser.add_argument('--tau',type=int,default=0,help='slippery step tau')
parser.add_argument('--dataset', type=str, default="cifar10", help="choose from cifar10,svhn,cifar100,mnist")
parser.add_argument('--random',type=bool,default=True,help="whether to initiat adversarial sample with random noise")
parser.add_argument('--omega',type=float,default=0.0,help="random sample parameter")
parser.add_argument('--dynamictau',type=bool,default=False,help='whether to use dynamic tau')
parser.add_argument('--depth',type=int,default=32,help='WRN depth')
parser.add_argument('--width-factor',type=int,default=10,help='WRN width factor')
parser.add_argument('--drop-rate',type=float,default=0.0, help='WRN drop rate')
parser.add_argument('--resume',type=str,default=None,help='whether to resume training')
parser.add_argument('--out-dir',type=str,default='./GAIRAT_result',help='dir of output')
parser.add_argument('--num-classes',type=int,default=10,help='num of classes')
parser.add_argument('--lr-schedule', default='piecewise', choices=['superconverge', 'piecewise', 'linear', 'piecewisesmoothed', 'piecewisezoom', 'onedrop', 'multipledecay', 'cosine'])
parser.add_argument('--lr-max', default=0.1, type=float)
parser.add_argument('--lr-one-drop', default=0.01, type=float)
parser.add_argument('--lr-drop-epoch', default=100, type=int)
parser.add_argument('--beta',type=float, default=-1.0, help='parameter for join loss')
parser.add_argument('--beta_max',type=float, default=100.0, help='max beta')
parser.add_argument('--beta_schedule', default='fixed', choices=['linear', 'piecewise', 'fixed'])
parser.add_argument('--loss_function', default='Tanh', choices=['Discrete','Sigmoid','Tanh'])
parser.add_argument('--begin_epoch', type=int, default=60, help='when to use GAIRLoss')

args = parser.parse_args()

# settings
seed = args.seed
momentum = args.momentum
weight_decay = args.weight_decay
depth = args.depth
width_factor = args.width_factor
drop_rate = args.drop_rate
resume = args.resume
out_dir = args.out_dir

torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

if args.net == "smallcnn_2_2":
    model = SmallCNN_2_2().cuda()
    net = "smallcnn_2_2"
if args.net == "smallcnn_4_3":
    model = SmallCNN_4_3().cuda()
    net = "smallcnn_4_3"
if args.net == "smallcnn_8_4":
    model = SmallCNN_8_4().cuda()
    net = "smallcnn_8_4"
if args.net == "smallcnn":
    model = SmallCNN().cuda()
    net = "smallcnn"
if args.net == "resnet18":
    model = ResNet18().cuda()
    net = "resnet18"
if args.net == "preactresnet18":
    model = PreActResNet18().cuda()
if args.net == "WRN":
    model = Wide_ResNet_Madry(depth=depth, num_classes=10, widen_factor=width_factor, dropRate=drop_rate).cuda()
    net = "WRN{}-{}-dropout{}".format(depth,width_factor,drop_rate)

model = torch.nn.DataParallel(model)
optimizer = optim.SGD(model.parameters(), lr=args.lr_max, momentum=momentum, weight_decay=weight_decay)

if args.lr_schedule == 'superconverge':
    lr_schedule = lambda t: np.interp([t], [0, args.epochs * 2 // 5, args.epochs], [0, args.lr_max, 0])[0]
elif args.lr_schedule == 'piecewise':
    def lr_schedule(t):
        if args.epochs >= 110:
            if t / args.epochs < 0.5:
                return args.lr_max
            elif t / args.epochs < 0.75:
                return args.lr_max / 10.
            elif t / args.epochs < (11/12):
                return args.lr_max / 100.
            else:
                return args.lr_max / 200.
        else:
            if t / args.epochs < 0.3:
                return args.lr_max
            elif t / args.epochs < 0.6:
                return args.lr_max / 10.
            else:
                return args.lr_max / 100.
elif args.lr_schedule == 'linear':
    lr_schedule = lambda t: np.interp([t], [0, args.epochs // 3, args.epochs * 2 // 3, args.epochs], [args.lr_max, args.lr_max, args.lr_max / 10, args.lr_max / 100])[0]
elif args.lr_schedule == 'onedrop':
    def lr_schedule(t):
        if t < args.lr_drop_epoch:
            return args.lr_max
        else:
            return args.lr_one_drop
elif args.lr_schedule == 'multipledecay':
    def lr_schedule(t):
        return args.lr_max - (t//(args.epochs//10))*(args.lr_max/10)
elif args.lr_schedule == 'cosine': 
    def lr_schedule(t): 
        return args.lr_max * 0.5 * (1 + np.cos(t / args.epochs * np.pi))


if not os.path.exists(out_dir):
    os.makedirs(out_dir)

def save_checkpoint(state, checkpoint=out_dir, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)

start_epoch = 0

# Resume 
title = 'GAIRAT'
best_acc = 0
if resume:
    # resume directly point to checkpoint.pth.tar
    print ('==> GAIRAT Resuming from checkpoint ..')
    print(resume)
    assert os.path.isfile(resume)
    out_dir = os.path.dirname(resume)
    checkpoint = torch.load(resume)
    start_epoch = checkpoint['epoch']
    best_acc = checkpoint['test_pgd20_acc']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    logger_test = Logger(os.path.join(out_dir, 'log_results.txt'), title=title, resume=True)
else:
    print('==> GAIRAT')
    logger_test = Logger(os.path.join(out_dir, 'log_results.txt'), title=title)
    logger_test.set_names(['Epoch', 'Natural Test Acc', 'PGD20 Acc'])


def train(epoch, model, train_loader, optimizer, beta):
    
    lr = 0
    num_data = 0
    train_robust_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.cuda(), target.cuda()
        loss = 0
        x_adv, pgd_steps = attack.pgd(model,data,target,args.epsilon,args.step_size,args.num_steps,loss_fn="cent",category="Madry",rand_init=True)

        model.train()
        lr = lr_schedule(epoch + 1)
        optimizer.param_groups[0].update(lr=lr)
        optimizer.zero_grad()
        
        logit = model(x_adv)
        pred = logit.max(1, keepdim=True)[1]

        if (epoch + 1) >= args.begin_epoch:
            pgd_steps = pgd_steps.cuda() 
            loss = GAIRLoss(logit, target, args.num_steps, pgd_steps, beta, loss_fn=args.loss_function, category="AT")
        else:
            loss = nn.CrossEntropyLoss(reduce="mean")(logit, target)
        train_robust_loss += loss.item() * len(x_adv)
        
        loss.backward()
        optimizer.step()
        
        num_data += len(data)

    train_robust_loss = train_robust_loss / num_data

    return train_robust_loss, lr


def eval_test(model, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_accuracy = correct / len(test_loader.dataset)
    return test_accuracy


def adjust_beta(epoch):
    if args.epochs >= 110:
        beta = args.beta_max
        if args.beta_schedule == 'linear':
            if epoch >= 60:
                beta = args.beta_max - (epoch/args.epochs) * (args.beta_max - args.beta)
        elif args.beta_schedule == 'piecewise':
            if epoch >= 60:
                beta = args.beta
            elif epoch >= 90:
                beta = args.beta-1.0
            elif epoch >= 110:
                beta = args.beta-1.5
        elif args.beta_schedule == 'fixed':
            if epoch >= 60:
                beta = args.beta
    else:
        beta = args.beta_max
        if args.beta_schedule == 'linear':
            if epoch >= 30:
                beta = args.beta_max - (epoch/args.epochs) * (args.beta_max - args.beta)
        elif args.beta_schedule == 'piecewise':
            if epoch >= 30:
                beta = args.beta
            elif epoch >= 60:
                beta = args.beta-2.0
        elif args.beta_schedule == 'fixed':
            if epoch >= 30:
                beta = args.beta
    return beta

# setup data loader
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
])

if args.dataset == "cifar10":
    trainset = torchvision.datasets.CIFAR10(root='~/dataset/cifar-10', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='~/dataset/cifar-10', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
if args.dataset == "svhn":
    trainset = torchvision.datasets.SVHN(root='~/dataset/SVHN', split='train', download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    testset = torchvision.datasets.SVHN(root='~/dataset/SVHN', split='test', download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
if args.dataset == "mnist":
    trainset = torchvision.datasets.MNIST(root='~/dataset/MNIST', train=True, download=True, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=1,pin_memory=True)
    testset = torchvision.datasets.MNIST(root='~/dataset/MNIST', train=False, download=True, transform=transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=1,pin_memory=True)

## Training get started

test_nat_acc = 0
test_pgd20_acc = 0

for epoch in range(start_epoch, args.epochs):
    
    
    beta = adjust_beta(epoch + 1)
    
    # Adversarial training
    train_robust_loss, lr = train(epoch, model, train_loader, optimizer, beta)

    # Evalutions the same as DAT.
    _, test_nat_acc = attack.eval_clean(model, test_loader)
    _, test_pgd20_acc = attack.eval_robust(model, test_loader, perturb_steps=20, epsilon=0.031, step_size=0.031 / 4,loss_fn="cent", category="Madry", random=True)


    print(
        'Epoch: [%d | %d] | Learning Rate: %f | Natural Test Acc %.2f | PGD20 Test Acc %.2f |\n' % (
        epoch,
        args.epochs,
        lr,
        test_nat_acc,
        test_pgd20_acc)
        )
         
    logger_test.append([epoch + 1, test_nat_acc, test_pgd20_acc])
    
    # Save the best checkpoint
    if test_pgd20_acc > best_acc:
        best_acc = test_pgd20_acc
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'test_nat_acc': test_nat_acc, 
                'test_pgd20_acc': test_pgd20_acc,
                'optimizer' : optimizer.state_dict(),
            },filename='bestpoint.pth.tar')

    # Save the last checkpoint
    save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'test_nat_acc': test_nat_acc, 
                'test_pgd20_acc': test_pgd20_acc,
                'optimizer' : optimizer.state_dict(),
            })
    
logger_test.close()