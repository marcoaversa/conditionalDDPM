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

from utils.base import set_dataloaders, digital_repr, electron_repr, decrease_intensity, downsample, upsample_clone

def import_ls(
        name: str = 'ls_full', 
        batch_size: int = 32, 
        image_size: int = 128, 
        transform = None, 
        BG: int = 325,
        noise_threshold: float = 20.,
        data_path: str = './data/light_sheets',
        force_download: bool = False):
    
    mode = name.split('_')[-1]
    
    if not os.path.exists(os.path.join(data_path, f'X_{image_size}.pt')) or force_download:
    
        # Check if path exists, if yes delete it
        if os.path.exists(os.path.join(data_path, f'X_{image_size}.pt')):
            os.remove(os.path.join(data_path, f'X_{image_size}.pt'))
        if os.path.exists(os.path.join(data_path, f'Y_{image_size}.pt')):
            os.remove(os.path.join(data_path, f'Y_{image_size}.pt'))

        # Find Sequences
        positions, z_stacks, x_shifts = detect_sequence(image_size = image_size, BG = BG)

        # Stack sequences
        seq, delta_zs=[],[]
        for i in range(101-max(z_stacks)):
            names = [f'./Pos{p:02d}/img_channel000_position{p:03d}_time000000000_z{z_stacks[j]+i:03d}.tif' for j,p in enumerate(positions)]
            seq.append([torch.clip(Tensor(tiff.imread(os.path.join(data_path, name)).astype(np.int16)),0,None) for name in names])
            delta_zs.append([z_stacks[j]+i - z_stacks[0] for j,p in enumerate(positions[1:])])

        # Tile Images
        n_shifts = len(positions)
        X, Y, Delta_Z= [], [], []
                
        seq = [torch.stack(s) for s in seq] 
        Y = [x[0] for x in seq]

        seq = torch.stack(seq)
        b,c,*_ = seq.shape
        tiles=[]
        for x in tqdm(seq): 
            x = torch.stack([img[:,(x_shifts[-1]-x_shifts[n]):-x_shifts[n]] if n>0 else img[:,x_shifts[-1]:] for n,img in enumerate(x)])
            for n in range(n_shifts-1):
                tiles.append(tile_multichannel_images(x, image_size))

        # Stack tiles
        X = torch.cat(tiles)
        
        # Keep first image (less noisy) as target and next one as input
        X = X[:,1:]        
        Y = X[:,0][:,None]
        
        # Permute images
        indices=torch.randperm(len(X))
        X = X[indices, None]
        Y = Y[indices, None] # Y.shape = (B,C,Steps,H,W)
        
        # Remove noisy images
        means = Tensor([img.mean().item() for img in Y])
        indices = means > noise_threshold
        Y = Y[indices]
        X = X[indices]

        # Save dataset
        torch.save(X, os.path.join(data_path, f'X_{image_size}.pt'))
        torch.save(Y, os.path.join(data_path, f'Y_{image_size}.pt'))
        print(f"\nDataset containes {len(X)} tiles")
        
    else:
        # Load dataset
        X = torch.load(os.path.join(data_path, f'X_{image_size}.pt'))
        Y = torch.load(os.path.join(data_path, f'Y_{image_size}.pt'))
        
    # Split dataset in train-valid set and apply transform
    train_size = int(len(X)*0.8)
    train_set = LightSheetsDataset( (Y[:train_size], X[:train_size]), mode=mode, transform=transform)
    valid_set = LightSheetsDataset( (Y[train_size:], X[train_size:]), mode=mode, transform=transform)
    train_loader, valid_loader = set_dataloaders([train_set, valid_set], batch_size)
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
            elif self.mode.startswith('aedpdown'):
                downsize = int(self.mode[-2:])
                y = torch.stack([downsample(digital_repr(decrease_intensity(electron_repr(img),0.5)),downsize) for img in x])
            elif self.mode.startswith('aedownup'):
                downsize = int(self.mode[-2:])
                y = torch.stack([upsample_clone(downsample(img,downsize)) for img in x])
                if downsize != 2:
                    x = torch.stack([downsample(img,downsize//2) for img in x])     
            
#                 crop_size=32
#                 x,y = Augmentations(crop_size=crop_size, crop=True)(x,y)
        
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

class Augmentations:
    def __init__(
            self, 
            crop_size: int = 32, 
            crop: bool = True, 
            flip: bool = True, 
            rotate: bool = True):
        self.crop_size = crop_size
        self.crop = crop
        self.flip = flip
        self.rotate = rotate

    def __call__(self, x, y=None):   
        # Random crop
        if self.crop and x.shape[-1] > self.crop_size:
            crop_indices = transforms.RandomCrop.get_params(x, output_size=(self.crop_size, self.crop_size))
            i, j, h, w = crop_indices
            x = F.crop(x, i, j, h, w)
            if y is not None:
                y = F.crop(y, i, j, h, w)

        # Random horizontal flipping
        if random.random() > 0.5 and self.flip:
            x = F.hflip(x)
            if y is not None:
                y = F.hflip(y)

        # Random vertical flipping
        if random.random() > 0.5 and self.flip:
            x = F.hflip(x)
            if y is not None:
                y = F.hflip(y)
            
        # Random rotate
        if random.random() > 0.5 and self.rotate:
            angles = 0,90,180,270
            angle = angles[int(random.random()*4)]
            x = F.rotate(x, angle)
            if y is not None:
                y = F.rotate(y, angle)

        return x,y
    
if __name__=='__main__':        
    train_loader, valid_loader = import_dataset(args)