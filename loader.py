import numpy as np
import json
import os
import torch
import sys
import torchvision
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
from torchvision.datasets import CIFAR10, CIFAR100
import math
import random



def corrupted_labels(targets, r = 0.4, noise_type='sym'):
    transition = {0: 0, 2: 0, 4: 7, 7: 7, 1: 1, 9: 1, 3: 5, 5: 3, 6: 6,
                       8: 8}  # class transition for asymmetric noise
    size = int(len(targets)*r)
    idx = list(range(len(targets)))
    random.shuffle(idx)
    noise_idx = idx[:size]
    noisy_label = []
    for i in range(len(targets)):
        if i in noise_idx:
            if noise_type == 'sym':
                noisy_label.append(random.randint(0,9))
            elif noise_type == 'asym':
                noisy_label.append(transition[targets[i]])
        else:
            noisy_label.append(targets[i])
    x = np.array(noisy_label)
    return x


class CIFAR10N(CIFAR10):
    """CIFAR10 Dataset.
    """
    def __init__(self, root, transform, noise_type, r):
        super(CIFAR10N, self).__init__(root, download=True)
        self.noise_targets = corrupted_labels(self.targets, r, noise_type)
        self.transform=transform

    def __getitem__(self, index):
        img, target, true_target = self.data[index], self.noise_targets[index], self.targets[index]
        img = self.data[index]
        img = Image.fromarray(img)

        im_1 = self.transform(img)

        return im_1, target, true_target, index




def corrupted_labels100(targets, r = 0.4, noise_type='sym'):
    size = int(len(targets)*r)
    idx = list(range(len(targets)))
    random.shuffle(idx)
    noise_idx = idx[:size]
    noisy_label = []
    for i in range(len(targets)):
        if i in noise_idx:
            if noise_type == 'sym':
                noisy_label.append(random.randint(0,99))
            elif noise_type == 'asym':
                noisy_label.append((targets[i]+1)%100)
        else:
            noisy_label.append(targets[i])
    x = np.array(noisy_label)
    return x


class CIFAR100N(CIFAR100):
    """CIFAR100 Dataset.
    """
    def __init__(self, root, transform, noise_type, r):
        super(CIFAR100N, self).__init__(root, download=True)
        self.noise_targets = corrupted_labels100(self.targets, r, noise_type)
        self.transform=transform

    def __getitem__(self, index):
        img, target, true_target = self.data[index], self.noise_targets[index], self.targets[index]
        img = self.data[index]
        img = Image.fromarray(img)


        im_1 = self.transform(img)


        return im_1, target, true_target, index
