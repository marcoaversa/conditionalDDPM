import os
import time
import sys
from tqdm import tqdm

import numpy as np

import torchvision
import torchvision.transforms as transforms

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torch.nn.functional import interpolate
from torchvision.transforms import GaussianBlur, CenterCrop
import torchvision.transforms.functional as F

import random

import tifffile as tiff

from typing import Union

import pyjetraw4ai_proto as jetraw4ai 

from .speckle_dataset import import_speckles
from .ls_dataset import import_ls
from .fiber_dataset import import_fiber

# import kornia
    
def import_dataset(data_name: str = 'MNIST', batch_size: int = 32, sum_from: int = 8, sum_to: int = 150, 
                   image_size = 128, import_timeseries = False, sum_every_n_steps: int = 1, 
                   seq_random: bool = True, seq_full: bool = False, force_download: bool = False):
    if data_name == 'MNIST':
        train_loader, valid_loader = import_mnist(batch_size)
    elif data_name == 'CelebA':
        train_loader, valid_loader = import_CelebA(batch_size)
    elif data_name == 'CIFAR10':
        train_loader, valid_loader = import_cifar10(batch_size)
    elif data_name == 'speckles':
        train_loader, valid_loader = import_speckles(sum_from = sum_from, 
                                                     sum_to = sum_to, 
                                                     batch_size = batch_size, 
                                                     import_timeseries = import_timeseries, 
                                                     sum_every_n_steps = sum_every_n_steps)
    
    if data_name.startswith('ls'):
        mode = data_name.split('_')[-1]
        train_loader, valid_loader = import_ls(mode = mode, 
                                               batch_size = batch_size,
                                               image_size = image_size,
                                               seq_random = seq_random,
                                               seq_full = seq_full,
                                               force_download = force_download)
    elif data_name.startswith('fiber'):
        mode = data_name.split('_')[-1]
        train_loader, valid_loader = import_fiber(mode, batch_size)
        
    return train_loader, valid_loader

def import_mnist(batch_size: int = 32):
    data_name = 'MNIST'
    data_path = './data'
    mu = (0.131,)
    sigma = (0.308,)
    channels=1
    dim_mults=(1,2,4)
    image_size=28
    train_set, valid_set = import_from_torchvision(data_name, data_path, mu, sigma)
    train_loader, valid_loader = set_dataloaders(train_set, valid_set, batch_size)
    return train_loader, valid_loader

def import_mnist_blurred(batch_size: int = 32):
    data_name = 'MNIST_blurred'
    data_path = './data'
    mu = (0.131,)
    sigma = (0.308,)
    channels=1
    dim_mults=(1,2,4)
    image_size=28
    train_set, valid_set = import_from_torchvision(data_name, data_path, mu, sigma)
    train_set = [(img, F.gaussian_blur(img, 11, (4.0, 4.0))) for (img,label) in train_set]
    valid_set = [(img, F.gaussian_blur(img, 11, (4.0, 4.0))) for (img,label) in valid_set]
    train_loader, valid_loader = set_dataloaders(train_set, valid_set, batch_size)
    return train_loader, valid_loader

def import_CelebA(batch_size: int = 32):
    data_name = 'CelebA'
    data_path = './data'
    mu = (0.5, 0.5, 0.5)
    sigma = (0.5, 0.5, 0.5)
    channels=3
    
    dataset_class = getattr(torchvision.datasets, data_name)

    train_transform, test_transform = set_transforms(mu,sigma)

    train_set = dataset_class(data_path, split='train', transform=train_transform, download=False)
    valid_set = dataset_class(data_path, split='valid', transform=test_transform, download=False)
    
    train_loader, valid_loader = set_dataloaders(train_set, valid_set, batch_size)
    return train_loader, valid_loader

def import_cifar10(batch_size: int = 32):
    data_name = 'CIFAR10'
    data_path = os.path.join('./data', data_name)
    mu = (0.49139968, 0.48215827 , 0.44653124) 
    sigma = (0.24703233, 0.24348505, 0.26158768)
    channels=3
    dim_mults=(1,2,4,8)
    image_size=32
    train_set, valid_set = import_from_torchvision(data_name, data_path, mu, sigma)
    train_loader, valid_loader = set_dataloaders(train_set, valid_set, batch_size)
    return train_loader, valid_loader

def set_transforms(
            mu: torch.Tensor = torch.Tensor([0.,0.,0.]),
            sigma: torch.Tensor = torch.Tensor([1.,1.,1.])):

    train_transform = transforms.Compose([
            transforms.ToTensor(),
#             transforms.RandomRotation(90),
#             transforms.RandomHorizontalFlip(),
#             transforms.RandomVerticalFlip(),
            transforms.Normalize(mu, sigma)
            ]) 

    test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mu, sigma)
            ]) 

    return train_transform, test_transform

def import_from_torchvision(
            data_name: str = 'MNIST', 
            data_path: str = './data', 
            mu: torch.Tensor = torch.Tensor([0.,]),
            sigma: torch.Tensor = torch.Tensor([1.,])):

    dataset_class = getattr(torchvision.datasets, data_name)

    train_transform, test_transform = set_transforms(mu,sigma)

    train_set = dataset_class(data_path, train = True, transform=train_transform, download=True)
    valid_set = dataset_class(data_path, train = False, transform=test_transform, download=True)

    return train_set, valid_set

if __name__=='__main__':        
    train_loader, valid_loader = import_dataset(args)