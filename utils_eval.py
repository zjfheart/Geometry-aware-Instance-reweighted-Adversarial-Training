"""
Utilities for PGD plus evaluation. 
Based on code from https://github.com/yaircarmon/semisup-adv
"""
import os
import numpy as np

from models.wrn_madry import Wide_ResNet_Madry
from models.resnet import *
from models.small_cnn import SmallCNN

import torch
from torch.nn import Sequential, Module


def get_model(name, num_classes=10, normalize_input=False):
    name_parts = name.split('-')
    if name_parts[0] == 'wrn':
        depth = int(name_parts[1])
        widen = int(name_parts[2])
        model = Wide_ResNet_Madry(
            depth=depth, num_classes=num_classes, widen_factor=widen)
        
    elif name_parts[0] == 'small':
        model = SmallCNN()
    elif name_parts[0] == 'resnet':
        model = ResNet18()
    else:
        raise ValueError('Could not parse model name %s' % name)

    if normalize_input:
        model = Sequential(NormalizeInput(), model)

    return model


class NormalizeInput(Module):
    def __init__(self, mean=(0.4914, 0.4822, 0.4465),
                 std=(0.2023, 0.1994, 0.2010)):
        super().__init__()

        self.register_buffer('mean', torch.Tensor(mean).reshape(1, -1, 1, 1))
        self.register_buffer('std', torch.Tensor(std).reshape(1, -1, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std

