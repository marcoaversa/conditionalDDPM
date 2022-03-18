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


import tifffile as tiff

import pyjetraw4ai_proto as jetraw4ai

    
def import_dataset(data_name: str = 'MNIST', batch_size: int = 32, sum_from: int = 0, sum_to: int = 50, 
                   image_size = 128, import_timeseries = False, sum_every_n_steps: int = 5, 
                   seq_random: bool = True, seq_full: bool = False, force_download: bool = False):
    if data_name == 'MNIST':
        train_loader, valid_loader = import_mnist(batch_size)
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
    BG = 443
    noise_threshold = 20.
    
    if not os.path.exists(os.path.join(data_path, f'X_{image_size}.pt')) or force_download:
    
        if os.path.exists(os.path.join(data_path, f'X_{image_size}.pt')):
            os.remove(os.path.join(data_path, f'X_{image_size}.pt'))
        if os.path.exists(os.path.join(data_path, f'Y_{image_size}.pt')):
            os.remove(os.path.join(data_path, f'Y_{image_size}.pt'))

        positions, z_stacks, x_shifts = detect_sequence(image_size = image_size, BG = BG)

        seq, delta_zs=[],[]
        for i in range(101-max(z_stacks)):
            names = [f'./Pos{p:02d}/img_channel000_position{p:03d}_time000000000_z{z_stacks[j]+i:03d}.tif' for j,p in enumerate(positions)]
            seq.append([torch.clip(Tensor(tiff.imread(os.path.join(data_path, name)).astype(np.int16))-BG,0,None) for name in names])
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

        X = X[:,1:]        
        Y = X[:,0][:,None]
            

        # Permute images
        indices=torch.randperm(len(X))
        X = X[indices, None]
        Y = Y[indices, None] # Y.shape = (B,C,Steps,H,W)

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
    
    def energy_norm(x):
        energy = x.mean()# --> x.shape = (C,H,W)
        x /= energy
        x_min = x.min()
        x_max = x.max()
        return (torch.clip(x - x_min,0,None)/(x_max-x_min)-0.5)*2
    
    transform = transforms.Compose([energy_norm,])

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
        
    if I_scale < 1.:
        
        params = jetraw4ai.Parameters(G,B,N,W)

        for i, img in tqdm(enumerate(imgs)):
            im_scaled = jetraw4ai.JetrawImage(img, params)
            imgs[i] = im_scaled.scale_exposure(factor).image_data

    return torch.Tensor(imgs)            

def diffusion(
        imgs, 
        factor: int =  1., 
        B: float = 99.6, 
        G: float = 2.2, 
        N: float = 3., 
        W: float = 2**16-1):
    
    
    assert factor <= 29400 or factor > 0, 'Intensity scale factor should be in the range (0.,29400.]'
    assert imgs.ndim == 3, 'Input image should be in the raw pattern format (B,H,W)'
    if isinstance(imgs, torch.Tensor):
        imgs = np.array(imgs).astype(np.float64)
    
    params = jetraw4ai.Parameters(G,B,N,W)

    for i, img in tqdm(enumerate(imgs)):
        jetraw_img = jetraw4ai.JetrawImage(img, params)
        el_repr = jetraw_img.electron_repr()
        el_repr.image_data += factor
        diffused_img = el_repr.replace_noise().digital_repr().image_data
        ratio = diffused_img.mean()/img.mean()
        diffused_img /= ratio
        imgs[i] = diffused_img
    return torch.Tensor(imgs)
        
def tile_image(image, image_size):
    """Image shape (H,W)"""
    return image.unfold(0,image_size,image_size).unfold(1,image_size,image_size).reshape(-1,image_size,image_size)


def tile_multichannel_images(image, image_size):
    """Image shape (C,H,W)"""
    c,*_ = image.shape
    image = image.unfold(1, image_size, image_size) #tile 1st axis
    image = image.unfold(2, image_size, image_size) #tile 2nd axis 
    image = image.reshape(c,-1,image_size,image_size) #stack tiles 
    image = image.permute(1,0,2,3) #reorder with channels on second axis
    return image


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

def set_dataloaders(train_set, valid_set, batch_size: int = 32):
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)
    return train_loader, valid_loader

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
            elif self.mode == 'firstlast':
                y = y[:,-1]
            elif self.mode == 'ae':
                y = x
            
        if self.transform:
            x = self.transform(x)
            if self.mode == 'full':
                for i in range(y.shape[1]):
                    y[:,i] = self.transform(y[:,i])
            else:
                y = self.transform(y)
            
#         if mode == 'DPSeqDarkening':
#             for i in range(1,X.shape[1]):
#                 factor = X[:,i].mean()/X[:,0].mean()
#                 print(f'Decreasing Intensity Step {i} --> Scaling factor = {factor:.2f}')
#                 X[:,i] = decrease_intensity(X[:,0].clone(), I_scale.item())
        
#         if mode == 'DPSeqDiffuse':
#             N_steps = 1000
#             ref_img = X[:,0].clone()
#             X = torch.zeros((X.shape[0],N_steps,X.shape[1],X.shape[2]))
#             for i in torch.linspace(100, 29000, N_steps):
#                 factor = int(i)
#                 print(f'Diffusing Step {i} --> Scaling factor = {factor}')
#                 X[:,i] = diffusion(X[:,0].clone(), factor)

#         if mode == 'ae':      
#             Y = X
#         else:
        x = x.type(torch.FloatTensor)        
        y = y.type(torch.FloatTensor)

        return x, y

    def __len__(self):
        return self.tensors[0].shape[0]

if __name__=='__main__':        
    train_loader, valid_loader = import_dataset(args)