CIFAR100_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

TINYIMAGENET_MEAN = (0.4802, 0.4481, 0.3975)

TINYIMAGENET_STD = (0.2302, 0.2265, 0.2262)
"""
   CIFAR-100, Tiny-ImageNet data loader
"""

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler


def fetch_dataloader(params):
    """
    Fetch and return train/dev dataloader with hyperparameters (params.subset_percent = 1.)
    """
    # using random crops and horizontal flip for train set
    if params.augmentation:
        train_transformer = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(0.5),  # randomly flip image horizontally
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)])
    
    # data augmentation can be turned off
    else:
        train_transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)])

    # transformer for dev set
    dev_transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)])

    # ************************************************************************************
    if params.dataset == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(root='./data/data-cifar100', train=True,
                                                download=True, transform=train_transformer)
        devset = torchvision.datasets.CIFAR100(root='./data/data-cifar100', train=False,
                                              download=True, transform=dev_transformer)

    # ************************************************************************************
    elif params.dataset == 'tinyimagenet':
        data_dir = './data/tiny-imagenet-200/'
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomRotation(20),
                transforms.RandomCrop(64, padding=8, padding_mode="edge"),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                transforms.Normalize(TINYIMAGENET_MEAN, TINYIMAGENET_STD),
            ]),
            'val': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(TINYIMAGENET_MEAN, TINYIMAGENET_STD),
            ])
        }
        train_dir = data_dir + 'train/'
        test_dir = data_dir + 'val/'
        trainset = torchvision.datasets.ImageFolder(train_dir, data_transforms['train'])
        devset = torchvision.datasets.ImageFolder(test_dir, data_transforms['val'])

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=params.batch_size,
                                              shuffle=True, num_workers=params.num_workers)

    devloader = torch.utils.data.DataLoader(devset, batch_size=params.batch_size,
                                            shuffle=False, num_workers=params.num_workers)

    return trainloader, devloader
