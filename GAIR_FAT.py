import os
import argparse
import torchvision
import torch.optim as optim
from torchvision import transforms
from models import *
from earlystop import GA_earlystop
from GAIR import GAIR
import numpy as np
import attack_generator as attack
from utils import Logger

parser = argparse.ArgumentParser(description='GAIR-FAT')
parser.add_argument('--epochs', type=int, default=120, metavar='N', help='number of epochs to train')
parser.add_argument('--weight-decay', '--wd', default=2e-4, type=float, metavar='W')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum')
parser.add_argument('--epsilon', type=float, default=0.031, help='perturbation bound')
parser.add_argument('--num-steps', type=int, default=10, help='maximum perturbation step K')
parser.add_argument('--step-size', type=float, default=0.007, help='step size')
parser.add_argument('--seed', type=int, default=3, metavar='S', help='random seed')
parser.add_argument('--net', type=str, default="WRN",help="decide which network to use,choose from smallcnn,resnet18,WRN")
parser.add_argument('--tau',type=int,default=0,help='slippery step tau')
parser.add_argument('--dataset', type=str, default="cifar10", help="choose from cifar10,svhn,cifar100,mnist")
parser.add_argument('--random',type=bool,default=True,help="whether to initiat adversarial sample with random noise")
parser.add_argument('--omega',type=float,default=0.0,help="random sample parameter")
parser.add_argument('--dynamictau',type=bool,default=True,help='whether to use dynamic tau')
parser.add_argument('--depth',type=int,default=32,help='WRN depth')
parser.add_argument('--width-factor',type=int,default=10,help='WRN width factor')
parser.add_argument('--drop-rate',type=float,default=0.0, help='WRN drop rate')
parser.add_argument('--resume',type=str,default=None,help='whether to resume training')
parser.add_argument('--out-dir',type=str,default='./GAIR_FAT_result',help='dir of output')
parser.add_argument('--lr-schedule', default='piecewise', choices=['superconverge', 'piecewise', 'linear', 'onedrop', 'multipledecay', 'cosine'])
parser.add_argument('--lr-max', default=0.1, type=float)
parser.add_argument('--lr-one-drop', default=0.01, type=float)
parser.add_argument('--lr-drop-epoch', default=100, type=int)
parser.add_argument('--Lambda',type=str, default='0.0', help='parameter for join loss')
parser.add_argument('--Lambda_max',type=float, default=float('inf'), help='max Lambda')
parser.add_argument('--Lambda_schedule', default='fixed', choices=['linear', 'piecewise', 'fixed'])
parser.add_argument('--weight_assignment_function', default='Tanh', choices=['Discrete','Sigmoid','Tanh'])
parser.add_argument('--begin_epoch', type=int, default=60, help='when to use GAIR')
args = parser.parse_args()

# Training settings
seed = args.seed
epsilon = args.epsilon
step_size = args.step_size
num_steps = args.num_steps
dynamictau = args.dynamictau
momentum = args.momentum
weight_decay = args.weight_decay
omega = args.omega
random = args.random
depth = args.depth
width_factor = args.width_factor
drop_rate = args.drop_rate
resume = args.resume
out_dir = args.out_dir

torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# Models and optimizer
if args.net == "smallcnn":
    model = SmallCNN().cuda()
    net = "smallcnn"
if args.net == "resnet18":
    model = ResNet18().cuda()
    net = "resnet18"
if args.net == "preactresnet18":
    model = PreActResNet18().cuda()
    net = "preactresnet18"
if args.net == "WRN":
    model = Wide_ResNet_Madry(depth=depth, num_classes=10, widen_factor=width_factor, dropRate=drop_rate).cuda()
    net = "WRN{}-{}-dropout{}".format(depth,width_factor,drop_rate)

model = torch.nn.DataParallel(model)
optimizer = optim.SGD(model.parameters(), lr=args.lr_max, momentum=momentum, weight_decay=weight_decay)

# Learning schedules
if args.lr_schedule == 'superconverge':
    lr_schedule = lambda t: np.interp([t], [0, args.epochs * 2 // 5, args.epochs], [0, args.lr_max, 0])[0]
elif args.lr_schedule == 'piecewise':
    def lr_schedule(t):
        if args.epochs >= 110:
            # Train Wide-ResNet
            if t / args.epochs < 0.5:
                return args.lr_max
            elif t / args.epochs < 0.75:
                return args.lr_max / 10.
            elif t / args.epochs < (11/12):
                return args.lr_max / 100.
            else:
                return args.lr_max / 200.
        else:
            # Train ResNet
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

# Store path
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# Save checkpoint
def save_checkpoint(state, checkpoint=out_dir, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    
# Get adversarially robust network
def train(epoch, model, train_loader, optimizer, tau, Lambda):
    
    lr = 0
    num_data = 0
    train_robust_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):

        loss = 0
        data, target = data.cuda(), target.cuda()

        # Get adversarial data and geometry value
        output_adv, output_target, output_natural, count, Kappa = GA_earlystop(model,data,target,step_size=args.step_size,epsilon=args.epsilon,perturb_steps=args.num_steps,tau=tau,type="fat",random=random,omega=omega)


        model.train()
        lr = lr_schedule(epoch + 1)
        optimizer.param_groups[0].update(lr=lr)
        optimizer.zero_grad()
        
        logit = model(output_adv)

        if (epoch + 1) >= args.begin_epoch:
            Kappa = Kappa.cuda()
            loss = nn.CrossEntropyLoss(reduce=False)(logit, target)
            # Calculate weight assignment according to geometry value
            normalized_reweight = GAIR(args.num_steps, Kappa, Lambda, args.weight_assignment_function)
            loss = loss.mul(normalized_reweight).mean()
        else:
            loss = nn.CrossEntropyLoss(reduce="mean")(logit, output_target)
        train_robust_loss += loss.item() * len(output_adv)
        
        loss.backward()
        optimizer.step()
        
        num_data += len(data)

    train_robust_loss = train_robust_loss / num_data

    return train_robust_loss, lr

# Adjust tau
def adjust_tau(epoch, dynamictau):
    tau = args.tau
    if dynamictau:
        if epoch <= 40:
            tau = 0
        elif epoch <= 70:
            tau = 3    
        else:
            tau = 6
    return tau

def adjust_Lambda(epoch):
    Lam = float(args.Lambda)
    if args.epochs >= 110:
        # Train Wide-ResNet
        Lambda = args.Lambda_max
        if args.Lambda_schedule == 'linear':
            if epoch >= 60:
                Lambda = args.Lambda_max - (epoch/args.epochs) * (args.Lambda_max - Lam)
        elif args.Lambda_schedule == 'piecewise':
            if epoch >= 60:
                Lambda = Lam
            elif epoch >= 90:
                Lambda = Lam-1.0
            elif epoch >= 110:
                Lambda = Lam-1.5
        elif args.Lambda_schedule == 'fixed':
            if epoch >= 60:
                Lambda = Lam
    else:
        # Train ResNet
        Lambda = args.Lambda_max
        if args.Lambda_schedule == 'linear':
            if epoch >= 30:
                Lambda = args.Lambda_max - (epoch/args.epochs) * (args.Lambda_max - Lam)
        elif args.Lambda_schedule == 'piecewise':
            if epoch >= 30:
                Lambda = Lam
            elif epoch >= 60:
                Lambda = Lam-2.0
        elif args.Lambda_schedule == 'fixed':
            if epoch >= 30:
                Lambda = Lam
    return Lambda

# Setup data loader
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
])

if args.dataset == "cifar10":
    trainset = torchvision.datasets.CIFAR10(root='./data/cifar-10', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data/cifar-10', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
if args.dataset == "svhn":
    trainset = torchvision.datasets.SVHN(root='./data/SVHN', split='train', download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    testset = torchvision.datasets.SVHN(root='./data/SVHN', split='test', download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
if args.dataset == "mnist":
    trainset = torchvision.datasets.MNIST(root='./data/MNIST', train=True, download=True, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=1,pin_memory=True)
    testset = torchvision.datasets.MNIST(root='./data/MNIST', train=False, download=True, transform=transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=1,pin_memory=True)

# Resume 
title = 'GAIR-FAT'
best_acc = 0
start_epoch = 0
if resume:
    # Resume directly point to checkpoint.pth.tar
    print ('==> GAIR-FAT Resuming from checkpoint ..')
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
    print('==> GAIR-FAT')
    logger_test = Logger(os.path.join(out_dir, 'log_results.txt'), title=title)
    logger_test.set_names(['Epoch', 'Natural Test Acc', 'PGD20 Acc'])

## Training get started
test_nat_acc = 0
test_pgd20_acc = 0

for epoch in range(start_epoch, args.epochs):

    # Get tau and lambda
    tau = adjust_tau(epoch + 1, dynamictau)
    Lambda = adjust_Lambda(epoch + 1)
    
    # Adversarial training
    train_robust_loss, lr = train(epoch, model, train_loader, optimizer, tau, Lambda)

    # Evalutions similar to DAT.
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