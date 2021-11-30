import os

import numpy as np

import torchvision
import torchvision.transforms as transforms

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

args = dict(data_name= 'MNIST', data_path= './data', batch_size= 32)
    
def import_dataset(data_name: str = 'MNIST', batch_size: str = 32):
    if data_name == 'MNIST':
        train_loader, valid_loader, image_size, channels, dim_mults = import_mnist(batch_size)
    elif data_name == 'CIFAR10':
        train_loader, valid_loader, image_size, channels, dim_mults = import_cifar10(batch_size)
    elif data_name == 'speckles':
        train_loader, valid_loader, image_size, channels, dim_mults = import_speckles(sum_from = 0, sum_to = 50, batch_size = batch_size)

    return train_loader, valid_loader, image_size, channels, dim_mults

def import_mnist(batch_size: int = 32):
    data_name = 'MNIST'
    data_path = './data'
    mu = (0.131,)
    sigma = (0.308,)
    image_size=28
    channels=1
    dim_mults=(1,2,4)
    train_set, valid_set = import_from_torchvision(data_name, data_path, mu, sigma)
    train_loader, valid_loader = set_dataloaders(train_set, valid_set, batch_size)
    return train_loader, valid_loader, image_size, channels, dim_mults

def import_cifar10(batch_size: int = 32):
    data_name = 'CIFAR10'
    data_path = os.path.join('./data', data_name)
    mu = (0.49139968, 0.48215827 , 0.44653124) 
    sigma = (0.24703233, 0.24348505, 0.26158768)
    image_size=32
    channels=3
    dim_mults=(1,2,4,8)
    train_set, valid_set = import_from_torchvision(data_name, data_path, mu, sigma)
    train_loader, valid_loader = set_dataloaders(train_set, valid_set, batch_size)
    return train_loader, valid_loader, image_size, channels, dim_mults

def import_speckles(sum_from: int = 0, sum_to: int = 10, batch_size: int = 32):
    data_path = './data/speckles'
    mu = (0,)
    sigma = (1,)
    image_size = 28
    channels = 1
    dim_mults = (1,2,4)
    X = np.load(os.path.join(data_path, 'targets.npz'))['arr_0']
    y = np.load(os.path.join(data_path, '80.0TMFPs.npz'))['arr_0'][:,:,:,sum_from:sum_to].sum(axis=3)[:,np.newaxis]
    # y = np.load(os.path.join(data_path, '80.0TMFPs.npz'))['arr_0']
    train_size = int(len(X)*0.8)

    train_transform, test_transform = set_transforms(mu,sigma)

    train_set = CustomTensorDataset( (X[:train_size], y[:train_size]), transform=train_transform )
    valid_set = CustomTensorDataset( (X[train_size:], y[train_size:]), transform=test_transform )
    train_loader, valid_loader = set_dataloaders(train_set, valid_set, batch_size)
    return train_loader, valid_loader, image_size, channels, dim_mults

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

def set_transforms(
            mu: torch.Tensor = torch.Tensor([0.,0.,0.]),
            sigma: torch.Tensor = torch.Tensor([1.,1.,1.])):
        
    train_transform = transforms.Compose([
            # transforms.RandomRotation(90),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mu, sigma)
            ]) 

    test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mu, sigma)
            ]) 

    return train_transform, test_transform

def set_dataloaders(train_set, valid_set, batch_size: int = 32):
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)
    return train_loader, valid_loader

class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms."""
    def __init__(self, tensors, transform=None):
        assert all(tensors[0].shape[0] == tensor.shape[0] for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]
        y = self.tensors[1][index]        
        
        if self.transform:
            x = self.transform(x)
            y = Tensor(y)

        x = x.type(torch.FloatTensor)        
        y = y.type(torch.FloatTensor)

        return x, y

    def __len__(self):
        return self.tensors[0].shape[0]

if __name__=='__main__':        
    train_loader, valid_loader = import_dataset(args)




