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

# import kornia
    
def import_dataset(data_name: str = 'MNIST', batch_size: int = 32, sum_from: int = 0, sum_to: int = 50, 
                   image_size = 128, import_timeseries = False, sum_every_n_steps: int = 5, 
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

def import_speckles(sum_from: int = 0, sum_to: int = 10, batch_size: int = 32, 
                    import_timeseries = False, sum_every_n_steps = 5):
    data_path = './data/speckles'
    mu = (0,)
    sigma = (1,)
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
            y = np.load(os.path.join(data_path, '80.0TMFPs.npz'))['arr_0'][:,:,:,sum_from][:,np.newaxis]
            for t in tqdm(range(sum_every_n_steps+sum_from,sum_to)[::sum_every_n_steps]):
                sum_to_t = np.load(os.path.join(data_path, '80.0TMFPs.npz'))['arr_0'][:,:,:,sum_from:t].sum(axis=3)[:,np.newaxis]
                sum_to_t = (sum_to_t-sum_to_t.min())/(sum_to_t.max()-sum_to_t.min())
                y = np.concatenate((y,sum_to_t), axis=1)
            np.save(os.path.join(data_path, f'80.0TMFPs_from_{sum_from}_to_{sum_to}_every_{sum_every_n_steps}.npy'), y)
        mu = (y[:train_size].mean().item(),)
        sigma = (y[:train_size].std().item(),)
        
    else:
        y = np.load(os.path.join(data_path, '80.0TMFPs.npz'))['arr_0'][:,:,:,sum_from:sum_to].sum(axis=3)[:,np.newaxis]

    train_transform, test_transform = set_transforms(mu,sigma)

    train_set = SpeckleDataset( (X[:train_size], y[:train_size]), transform=train_transform , size = image_size )
    valid_set = SpeckleDataset( (X[train_size:], y[train_size:]), transform=test_transform , size = image_size )
    train_loader, valid_loader = set_dataloaders(train_set, valid_set, batch_size)
    return train_loader, valid_loader


def detect_sequence(data_path: str = './data/light_sheets', image_size: int = 128, BG: int = 443):
    """Detect Light sheets sequence along z_stack during shift"""
    
    loss_fn = lambda x,y: torch.sqrt(((x-y)**2).mean())
    norm = lambda img: (img-img.min())/(img.max()-img.min())

    img_ref_path = os.path.join(data_path,f'./Pos{1:02d}/img_channel000_position{1:03d}_time000000000_z{4:03d}.tif')
    img_ref = Tensor(tiff.imread(img_ref_path).astype(np.int32))[None,None]    
    img_ref = torch.clip(img_ref-BG,0,None)
    energy = [img_ref.mean(),]
    img_ref = norm(img_ref)
#     img_ref = CenterCrop(400)(img_ref)
    img_ref = GaussianBlur(5, sigma=(5.0, 5.0))(img_ref)

    zs,xs,p = [4,],[0,],[1,]
    for pos in range(2,18):
        z_close, x_close = 0,1
        loss = 10000
        for z in range(np.clip(4*pos-2,0,None), 4*pos+2):
            img_path = os.path.join(data_path,f'./Pos{pos:02d}/img_channel000_position{pos:03d}_time000000000_z{z:03d}.tif')
            img = Tensor(tiff.imread(img_path).astype(np.int32))[None,None]    
            img = torch.clip(img.clone()-BG,0,None) 
            energy_temp = img.mean()
#             energy.append(img.mean())
            img = norm(img)
#             img = CenterCrop(400)(img)
            img = GaussianBlur(5, sigma=(5.0, 5.0))(img)
            for shift in range(-50,50):
                shift =  np.clip(shift+38*(pos-1),1,None)
                temp = loss_fn(img.squeeze()[:,:-shift], img_ref.squeeze()[:,shift:])
                if temp < loss:
                    loss = temp
                    z_close = z
                    x_close = shift    

        if z_close < zs[-1] or x_close < xs[-1] or image_size > img.shape[-1]-x_close:
            break
            
        print(f'Position {pos} --> z_shift {z_close} x_shift {x_close} energy {energy[-1]:.2f}')

        zs.append(z_close)
        xs.append(x_close)
        p.append(pos)
        energy.append(energy_temp)
        
    return p, zs, xs


def import_ls(mode: str = 'seq', batch_size: int = 32, image_size: int = 128, 
              seq_random: bool = True, seq_full: bool = False, force_download: bool = False):
    """Lightsheets dataset
       
       Args:
           -mode = different X,Y light sheet pairs
               -full = Y first stack image, X all diffusions for Y. On X we have an additional map with the level of depth
               -ae = X is equal to Y
    """
    data_path = './data/light_sheets'
    channels = 1
    dim_mults = (1,2,4,8)
#     BG = 443
    BG = 325
    noise_threshold = 20.
    transform = None
    
    if not os.path.exists(os.path.join(data_path, f'X_{image_size}.pt')) or force_download:
    
        if os.path.exists(os.path.join(data_path, f'X_{image_size}.pt')):
            os.remove(os.path.join(data_path, f'X_{image_size}.pt'))
        if os.path.exists(os.path.join(data_path, f'Y_{image_size}.pt')):
            os.remove(os.path.join(data_path, f'Y_{image_size}.pt'))

        positions, z_stacks, x_shifts = detect_sequence(image_size = image_size, BG = BG)

        seq, delta_zs=[],[]
        for i in range(101-max(z_stacks)):
            names = [f'./Pos{p:02d}/img_channel000_position{p:03d}_time000000000_z{z_stacks[j]+i:03d}.tif' for j,p in enumerate(positions)]
            seq.append([torch.clip(Tensor(tiff.imread(os.path.join(data_path, name)).astype(np.int16)),0,None) for name in names])
            delta_zs.append([z_stacks[j]+i - z_stacks[0] for j,p in enumerate(positions[1:])])

        n_shifts = len(positions)
        
        X, Y, Delta_Z= [], [], []
                        
        print('\nTiling images')
                
        seq = [torch.stack(s) for s in seq] 
        Y = [x[0] for x in seq]

        seq = torch.stack(seq)
        b,c,*_ = seq.shape
        tiles=[]
        for x in tqdm(seq): 
            x = torch.stack([img[:,(x_shifts[-1]-x_shifts[n]):-x_shifts[n]] if n>0 else img[:,x_shifts[-1]:] for n,img in enumerate(x)])
            for n in range(n_shifts-1):
                tiles.append(tile_multichannel_images(x, image_size))

        X = torch.cat(tiles)
        
#         print('Stack Denoised Images')
#         X_denoised = X.clone()
#         for x in tqdm(X):
#             X_denoised[i]= TVDenoising(x.clone()).detach()
            
#         conc = torch.cat([X[:,None],X_denoised[:,None]], dim=1)
#         X = X[:,:,1:]        
#         Y = X[:,:,0][:,:,None]
        
        X = X[:,1:]        
        Y = X[:,0][:,None]
        
        # Permute images
        indices=torch.randperm(len(X))
        X = X[indices, None]
        Y = Y[indices, None] # Y.shape = (B,C,Steps,H,W)
#         X = X[indices]
#         Y = Y[indices] # Y.shape = (B,C,Steps,H,W)
        

        print('Saving Tiles')
        # Remove noisy images
        means = Tensor([img.mean().item() for img in Y])
        indices = means > noise_threshold
        Y = Y[indices]
        X = X[indices]

        torch.save(X, os.path.join(data_path, f'X_{image_size}.pt'))
        torch.save(Y, os.path.join(data_path, f'Y_{image_size}.pt'))
        print(f"\nDataset containes {len(X)} tiles")
        
    else:
        print("Loading Dataset")
        X = torch.load(os.path.join(data_path, f'X_{image_size}.pt'))
        Y = torch.load(os.path.join(data_path, f'Y_{image_size}.pt'))
        
    train_size = int(len(X)*0.8)
    
#     def energy_norm(x):
#         energy = x.mean()# --> x.shape = (C,H,W)
#         x /= energy
#         x = (x-x.mean())/x.std()
#         return x
# #         x_min = x.min()
# #         x_max = x.max()
# #         return (torch.clip(x - x_min,0,None)/(x_max-x_min)-0.5)*2
    
#     def energy_norm(x):
#         BG=325
#         #Mean and std computed after BG removed
#         mu=163.0387
# #         sigma=3.2148
#         sigma=52.404
#         return (x-BG-mu)/sigma
# #         return (x/2**16-1)-0.5
    
#     def norm(x):
#         vmin = 330
#         vmax = 10852
#         x = (x-vmin)/(vmax-vmin)
#         return x
    
    mu = 473.93
    sigma = 50.44*5 # normalize to 5 standard deviations
#     mu = 0.0137
#     sigma = 0.0048

#     def clip(x):
#         return torch.clip(x,-5,5)
    
#     transform = transforms.Compose([
#                                     transforms.Normalize(mu, sigma),
# #                                     clip
#                                     ])

    train_set = LightSheetsDataset( (Y[:train_size], X[:train_size]), mode=mode, transform=transform)
    valid_set = LightSheetsDataset( (Y[train_size:], X[train_size:]), mode=mode, transform=transform)
    train_loader, valid_loader = set_dataloaders(train_set, valid_set, batch_size)
    print('Light Sheet data imported!')
    return train_loader, valid_loader


def decrease_intensity(
                imgs, 
                factor: float =  1., 
                B: float = 99.6, 
                G: float = 2.2, 
                N: float = 3., 
                W: float = 2**16-1):
    """
    Rescale the intensity of a raw image.
    REMARK: In the figure (B == c, G == z, N ==  sigma, W == max_value)

    Args:
        img (torch.Tensor,numpy.array): raw image
        factor (float): percentage of intensity rescaling, range (0.,1.]
        B (float): Black level
        G (float): Gain
        N (float): Standard deviation
        W (int): White Level

    Return:
        img_rescaled (torch.Tensor): Rescaled RGGB image    
    """

    assert factor <= 1. or factor > 0., 'Intensity scale factor should be in the range (0.,1.]'
    assert imgs.ndim == 3, 'Input image should be in the raw pattern format (B,H,W)'
    if isinstance(imgs, torch.Tensor):
        imgs = np.array(imgs).astype(np.float64)
        
    if factor < 1.:
        
        params = jetraw4ai.Parameters(G,B,N,W)

        for i, img in enumerate(imgs):
            im_scaled = jetraw4ai.JetrawImage(img, params)
            imgs[i] = im_scaled.scale_exposure(factor).image_data

    return torch.Tensor(imgs)            


class SpeckleDataset(Dataset):
    """TensorDataset with support of transforms."""
    def __init__(self, tensors, transform=None, size: int = 28):
        assert all(tensors[0].shape[0] == tensor.shape[0] for tensor in tensors)
        self.tensors = tensors
        self.transform = transform
        self.size = size

    def __getitem__(self, index):
        x = self.tensors[0][index].clone()
        y = self.tensors[1][index].clone()
        
        if self.transform:
            x = self.transform(x)
            y = Tensor(y).clone()
            y = (y-y.min())/(y.max()-y.min())
            
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
            

class AddGaussianNoise(object):
    def __init__(self,std):
        self.std = std
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
 

class LightSheetsDataset(Dataset):
    """TensorDataset with support of transforms."""
    def __init__(self, 
                 tensors, 
                 mode:str = 'full', 
                 transform=None):
        assert all(tensors[0].shape[0] == tensor.shape[0] for tensor in tensors)
        self.tensors = tensors
        self.transform = transform
        self.mode = mode

    def __getitem__(self, index):
        x = self.tensors[0][index].clone()
        y = self.tensors[1][index].clone()
    
        if self.mode != 'full':
            x = x[:,0]
            if self.mode == 'random':
                index=torch.randint(0,len(y[0])-1,(1,)).item()
                y = y[:,index]
            elif self.mode.startswith('step'):
                step = int(self.mode[-2:])
                y = y[:,step]
            elif self.mode == 'aedp':
                y = torch.stack([digital_repr(decrease_intensity(electron_repr(img),0.5)) for img in x])
            elif self.mode == 'aedpdown':
                downsize = int(self.mode[-2:])
                y = torch.stack([downsample(digital_repr(decrease_intensity(electron_repr(img),0.5)),downsize) for img in x])
            elif self.mode.startswith('aedownup'):
                downsize = int(self.mode[-2:])
                y = torch.stack([upsample_clone(downsample(img,downsize)) for img in x])
#                 y = torch.stack([AddGaussianNoise(img.std())(GaussianBlur((5,5),(2.0,2.0))(upsample_clone(downsample(img,downsize))[None,None])[0,0]) for img in x])
                if downsize != 2:
                    x = torch.stack([downsample(img,downsize//2) for img in x])     
            
                crop_size=32
                x,y = Augmentations(crop_size=crop_size, crop=True)(x,y)
        
        if self.transform:
            x = self.transform(x)
            if self.mode == 'full':
                for i in range(y.shape[1]):
                    y[:,i] = self.transform(y[:,i])
            else:
                y = self.transform(y)
        
        x = x.type(torch.FloatTensor)        
        y = y.type(torch.FloatTensor)

        return x, y

    def __len__(self):
        return self.tensors[0].shape[0]

if __name__=='__main__':        
    train_loader, valid_loader = import_dataset(args)