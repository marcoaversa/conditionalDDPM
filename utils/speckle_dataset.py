import os
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from torch import Tensor

import torchvision
import torchvision.transforms as transforms

from utils.base import set_dataloaders

def import_speckles(sum_from: int = 8, sum_to: int = 150, batch_size: int = 32, 
                    import_timeseries = False, sum_every_n_steps = 1, illumination='flat'):
    data_path = '/nfs/conditionalDDPM/data/speckles'
    mu = (0,)
    sigma = (4.3736e+08,)
    image_size = 28
    channels = 1
    dim_mults = (1,2,4)
    sum_from = 6 if sum_from < 6 else sum_from # First 5 timesteps don't have signal
    X = np.load(os.path.join(data_path, 'targets.npz'))['arr_0']
    train_size = int(len(X)*0.8)
    if import_timeseries:
        if f'80.0TMFPs_from_{sum_from}_to_{sum_to}_every_{sum_every_n_steps}.npy' in os.listdir(data_path):
            y = np.load(os.path.join(data_path, f'80.0TMFPs_from_{sum_from}_to_{sum_to}_every_{sum_every_n_steps}.npy'))
        else:
            print("Import timeseries\n")
            y = np.load(os.path.join(data_path, '80.0TMFPs.npz'))['arr_0'][:,:,:,sum_from][...,np.newaxis]
            for t in tqdm(range(sum_every_n_steps+sum_from,sum_to)[::sum_every_n_steps]):
                sum_to_t = np.load(os.path.join(data_path, '80.0TMFPs.npz'))['arr_0'][:,:,:,sum_from:t].sum(axis=3)[...,np.newaxis]
                sum_to_t = (sum_to_t-sum_to_t.min())/(sum_to_t.max()-sum_to_t.min())
                y = np.concatenate((y,sum_to_t), axis=-1)
            np.save(os.path.join(data_path, f'80.0TMFPs_from_{sum_from}_to_{sum_to}_every_{sum_every_n_steps}.npy'), y)
        mu = (y[:train_size].mean().item(),)
        sigma = (y[:train_size].std().item(),)
        
    else:
        name = '80.0TMFPs_flat.npz' if illumination == 'flat' else '80.0TMFPs.npz'
        y = np.load(os.path.join(data_path, name))['arr_0'][:,:,:,sum_from:sum_to].sum(axis=3)[...,np.newaxis]
        y = y[:1024]

        
    train_transform, test_transform = set_transforms(mu,sigma)

    train_set = SpeckleDataset( (X[:train_size], y[:train_size]), transform=train_transform , size = image_size )
    valid_set = SpeckleDataset( (X[train_size:], y[train_size:]), transform=test_transform , size = image_size )
    train_loader, valid_loader = set_dataloaders([train_set, valid_set], batch_size)
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

class SpeckleDataset(Dataset):
    """TensorDataset with support of transforms."""
    def __init__(self, tensors, transform=None, size: int = 28):
        assert all(tensors[0].shape[0] == tensor.shape[0] for tensor in tensors)
        self.tensors = tensors
        self.transform = transform
        self.size = size

    def __getitem__(self, index):
        x = self.tensors[0][index].copy()
        y = self.tensors[1][index].copy()
        
        if self.transform:
            y = self.transform(y)
            x = transforms.ToTensor()(x)

        x = self.upsample_img(x)
        y = self.upsample_img(y)

        x = x.type(torch.FloatTensor)        
        y = y.type(torch.FloatTensor)

        return x, y

    def __len__(self):
        return self.tensors[0].shape[0]
    
    def upsample_img(self, imgs: torch.Tensor):
        c,h,w = imgs.shape
        if self.size == h:
            return imgs
        elif self.size > h:
            return torch.squeeze(interpolate(imgs[None], size=(self.size,self.size)), dim=0)
        elif self.size < h:
            assert self.size < h, 'I haven t implemented yet the downsampling'