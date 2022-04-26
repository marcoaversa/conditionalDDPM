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
    sigma = 50.44
#     mu = 0.0137
#     sigma = 0.0048

    def clip(x):
        return torch.clip(x,-5,5)
    
    transform = transforms.Compose([
                                    transforms.Normalize(mu, sigma),
                                    clip
                                    ])

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

def replace_noise(
                imgs, 
                factor: float =  1., 
                B: float = 99.6, 
                G: float = 2.2, 
                N: float = 3., 
                W: float = 2**16-1):
    
    assert imgs.ndim == 3, 'Input image should be in the raw pattern format (B,H,W)'
    if isinstance(imgs, torch.Tensor):
        imgs = np.array(imgs).astype(np.float64)
        
    params = jetraw4ai.Parameters(G,B,N,W)
    
    for i, img in enumerate(imgs):
        im_scaled = jetraw4ai.JetrawImage(img, params)
        imgs[i] = im_scaled.replace_noise().image_data

    return torch.Tensor(imgs)
    
def diffusion(
        imgs, 
        factor: float =  1., 
        B: float = 99.6, 
        G: float = 2.2, 
        N: float = 3., 
        W: float = 2**16-1):
    
    
    assert factor <= 1. or factor >= 0, 'Intensity scale factor should be in the range (0.,29400.]'
    assert imgs.ndim == 3, 'Input image should be in the raw pattern format (B,H,W)'
    if isinstance(imgs, torch.Tensor):
        imgs = np.array(imgs).astype(np.float64)
    
    params = jetraw4ai.Parameters(G,B,N,W)

    for i, img in enumerate(imgs):        
        jetraw_img = jetraw4ai.JetrawImage(img, params)
        el_repr = jetraw_img.electron_repr()
        v_max = el_repr.image_data.max()
        v_max_el = int((W-B)/G)
        f = int(factor*(v_max_el-v_max))
        el_repr.image_data += f
        diffused_img = el_repr.replace_noise().digital_repr().image_data
        ratio = diffused_img.mean()/img.mean()
        diffused_img /= ratio
        imgs[i] = diffused_img
    return torch.Tensor(imgs)

def image_info(image: Union[np.ndarray, Tensor]):
    shape = image.shape
    if len(shape) == 3:
        if isinstance(image, Tensor):
            C, H, W = shape
        elif isinstance(image, np.ndarray):
            H, W, C = shape
        else:
            raise NotImplementedError("Expected torch.Tensor or np.ndarray")
    elif len(shape) == 2:
        H, W = shape
        C = 1
    else:
        raise NotImplementedError("Image shape is {shape}. It expected image with shape (H,W), (C,H,W) or (H,W,C)")

    return C, H, W

def image2patches(image: Union[np.ndarray, Tensor], patch_size: list, view_as_patches: bool = False):
    """
        Patch a 2D image in tiles.

        Args:
            image: 
                Image to patch. Image should be 2D. 
            patch_size:
                Final size of the tiles. 
            view_as_patches:
                Example for 2D image:
                    If set true -> return the final output is 4D (N_patch_x, N_patch_y, Patch_size_x, Patch_size_y)
                        image.shape = (100,100), patch_size = (2,2) -> patches.shape = (50,50,2,2)
                    If set False -> return the final output is 3D (N_patch, Patch_size_x, Patch_size_y)
                        image.shape = (100,100), patch_size = (2,2) -> patches.shape = (2500,2,2)
        Return:
            patches:
                Image splitted in patches        

    """

    C, H, W = image_info(image)
    assert C == 1, "Image needs to be 2D."
    p0, p1 = patch_size

    x_max = p0*(H//p0) # integer number for patches
    y_max = p1*(W//p1)
    
    image = image[0:x_max, 0:y_max]
    if view_as_patches:
        return image.reshape(H//p0,p0,W//p1,p1).swapaxes(1,2).reshape(-1,p0,p1)
    else:
        return image.reshape(H//p0,p0,W//p1,p1).swapaxes(1,2)


def patches2image(patches):
    """
        Return patches obtained from a 2D image into the original image.

        Args:
            patches: 
                Tiles to return into the 2D image. Shape = (N_patch_x, N_patch_y, Patch_size_x, Patch_size_y)
            Return:
                image:
                    2D image reconstructed from patches

    """
    
    assert len(patches.shape) == 4, "patches2image is compatible with image2patches, input patches should be in the shape (N_patch_x, N_patch_y, Patch_size_x, Patch_size_y)"
    n_h,n_w,p0,p1 = patches.shape
    return patches.reshape(n_h, n_w, p0,p1).swapaxes(1,2).reshape(n_h*p0,n_w*p1)

def downsample(image, downsample_size):
    if image.ndim == 2:
        image = image[None]
    assert image.ndim == 3, f'Image shape should be (C,H,W) instead of {image.shape}'
    patches = image2patches(image, patch_size= [downsample_size,downsample_size], view_as_patches = False)
    image = patches2image(patches.mean(-1).mean(-1)[None,None])[None]
    return image[0]

def inv_split_mosaic(imgs: torch.Tensor):

    C, H, W = imgs.shape
    
    mosaic = torch.zeros((H*2, W*2))
    imgs = [imgs[c] for c in range(C)]

    c = 0
    for i in range(2):
        for j in range(2):
            mosaic[i::2, j::2] = imgs[c] 
            c+=1

    return mosaic

def upsample_clone(image):
    image = image[None].repeat(4,1,1)
    image = inv_split_mosaic(image)
    return image

def upsample_interpolation(img):
    return interpolate(img[None,None], size=(2,2))[0,0]

# class TVDenoise(torch.nn.Module):
#     def __init__(self, noisy_image):
#         super(TVDenoise, self).__init__()
#         self.l2_term = torch.nn.MSELoss(reduction='mean')
#         self.regularization_term = kornia.losses.TotalVariation()
#         # create the variable which will be optimized to produce the noise free image
#         self.clean_image = torch.nn.Parameter(data=noisy_image.clone(), requires_grad=True)
#         self.noisy_image = noisy_image

#     def forward(self):
#         return self.l2_term(self.clean_image, self.noisy_image) + 0.0001 * self.regularization_term(self.clean_image)

#     def get_clean_image(self):
#         return self.clean_image

# def TVDenoising(image: torch.Tensor):
#     """Image shape = (B,C,W,H)"""
    
#     tv_denoiser = TVDenoise(image)
#     optimizer = torch.optim.SGD(tv_denoiser.parameters(), lr=0.1, momentum=0.9)

#     # run the optimization loop
#     num_iters = 500
#     for i in range(num_iters):
#         optimizer.zero_grad()
#         loss = tv_denoiser()
#         loss.backward()
#         optimizer.step()
        
#     return tv_denoiser.get_clean_image()
    
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
            
electron_repr = lambda x: (x - 99.6)/ 2.2
digital_repr = lambda x: (x * 2.2 + 99.6)

def decrease_intensity(image_data: torch.Tensor, factor: float):
    if not 0 <= factor <= 1:
        raise ValueError("factor must be between 0 and 1")

    image_data = electron_repr(image_data)
    if factor != 1:
        scaled_data = factor * image_data
        noise_var = (1 - factor) * torch.clip(scaled_data, 0, None) + \
                    (1 - factor ** 2) * (3.0 / 2.2) ** 2
        scaled_data = digital_repr(scaled_data + torch.normal(0, torch.sqrt(noise_var)))
        return torch.clip(scaled_data,0,2**16-1)
    else:
        return digital_repr(image_data)   

class AddGaussianNoise(object):
    def __init__(self,std):
        self.std = std
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
 
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