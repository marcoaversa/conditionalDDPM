import os
from xmlrpc.client import Boolean

from PIL import Image
import tifffile as tiff
import rawpy

import numpy as np

from scipy.ndimage import convolve

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from typing import Union

import dotphoton.base_cpp as DPcpp
import pyjetraw4ai_proto as jetraw4ai

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import time

class Timer(object):
    """Measure for a sequential of operations
    
    Example:
        with Timer('foo_stuff'):
            # do some foo
            # do some stuff
            # time.sleep(5)
    """        
    
    def __init__(self, name=None):
        self.name = name

    # __enter__ and __exit__ are for the with statement
    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print('[%s]' % self.name,)
        print('Elapsed: %s' % (time.time() - self.tstart))

def stack_dtype(var):
    if isinstance(var[0], Tensor):
        return torch.stack(var)
    elif isinstance(var[0], np.ndarray):
        return np.stack(var, axis=2)
    else:
        raise NotImplementedError("Expected torch.Tensor or np.ndarray")

def convolve_dtype(var: Union[torch.Tensor, np.ndarray], kernels: Union[torch.Tensor, np.ndarray], padding: bool = False):
    if isinstance(var, Tensor):
        assert var.ndim == 3, f'Image shape should be shape = (in_channels,i*H,i*W) '
        var = var.float() 
        if not padding:
            new_var = torch.zeros((var.shape[0],var.shape[1]-(kernels[0].shape[0]//2)*2, var.shape[2]-(kernels[0].shape[1]//2)*2))
        for k in range(kernels.shape[0]):
            kernel = kernels[k][None,None]
            img = var[k][None,None]
            if padding:
                var[k] = F.conv2d(input = img, weight = kernel, padding=((kernel.shape[-2]//2,kernel.shape[-1]//2)))
            else:
                new_var[k] = F.conv2d(input = img, weight = kernel)
        if padding:
            return var.squeeze()
        else:
            return new_var.squeeze()
            
    elif isinstance(var, np.ndarray):
        var = var.astype('float64')
        return convolve(var, kernels)
        
    else:
        raise NotImplementedError("Expected torch.Tensor or np.ndarray")
        

def np2torch(nparray):
    """Convert numpy array to torch tensor
       For array with more than 3 channels, it is better to use an input array in the format BxHxWxC

       Args:
           numpy array (ndarray) BxHxWxC
       Returns:
           torch tensor (tensor) BxCxHxW"""

    tensor = torch.Tensor(nparray)

    if tensor.ndim == 2:
        return tensor
    if tensor.ndim == 3:
        height, width, channels = tensor.shape
        if channels <= 3:  # Single image with more channels (HxWxC)
            return tensor.permute(2, 0, 1)

    if tensor.ndim == 4:  # More images with more channels (BxHxWxC)
        return tensor.permute(0, 3, 1, 2)

    return tensor



def torch2np(torchtensor):
    """Convert torch tensor to numpy array 
       For tensor with more than 3 channels or batch, it is better to use an input tensor in the format BxCxHxW

       Args:
           torch tensor (tensor) BxCxHxW
       Returns:
           numpy array (ndarray) BxHxWxC"""

    ndarray = torchtensor.detach().cpu().numpy().astype(np.float32)

    if ndarray.ndim == 3:  # Single image with more channels (CxHxW)
        channels, height, width = ndarray.shape
        if channels <= 3:
            return ndarray.transpose(1, 2, 0)

    if ndarray.ndim == 4:  # More images with more channels (BxCxHxW)
        return ndarray.transpose(0, 2, 3, 1)

    return ndarray


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
        raise NotImplementedError(f"Image shape is {shape}. It expected image with shape (H,W), (C,H,W) or (H,W,C)")

    return C, H, W

def load_image(path):
    file_type = path.split('.')[-1].lower()
    if file_type == 'dng':
        img = rawpy.imread(path).raw_image_visible
    elif file_type == 'tiff' or file_type == 'tif':
        img = np.array(tiff.imread(path))
    else:
        img = np.array(Image.open(path))
    return img.astype(np.float32)


def list_images_in_dir(path):
    IMAGE_FILE_TYPES = ['dng', 'png', 'tif', 'tiff']
    image_list = [os.path.join(path, img_name)
                  for img_name in sorted(os.listdir(path))
                  if img_name.split('.')[-1].lower() in IMAGE_FILE_TYPES]
    return image_list


def check_image_folder_consistency(images, masks):
    file_type_images = images[0].split('.')[-1].lower()
    file_type_masks = masks[0].split('.')[-1].lower()
    assert len(images) == len(masks), "images / masks length mismatch"
    for img_file, mask_file in zip(images, masks):
        img_name = img_file.split('/')[-1].split('.')[0]
        assert img_name in mask_file, f"image {img_file} corresponds to {mask_file}?"
        assert img_file.split('.')[-1].lower() == file_type_images, \
            f"image file {img_file} file type mismatch. Shoule be: {file_type_images}"
        assert mask_file.split('.')[-1].lower() == file_type_masks, \
            f"image file {mask_file} file type mismatch. Should be: {file_type_masks}"


def show_fig(images: Union[list, np.ndarray, torch.Tensor], title = None, ax_titles: list = (None,), colorbar = False, norm = True, figsize = (10,5)):


        if not isinstance(images, list):
            images = images.cpu() if isinstance(images, torch.Tensor) else images
            if norm:
                images = (images-images.min())/(images.max()-images.min())
            plt.figure()
            plt.imshow(images)
            if colorbar==True:
                plt.colorbar()
        else:    
            N = len(images)

            if norm:
                for i,img in enumerate(images):
                    images[i] = (img-img.min())/(img.max()-img.min())

            if N/4>1:
                c = 4
                r = N//4 
            else:
                c = N
                r = 1
                
            fig, axes = plt.subplots(r,c, figsize=figsize)
            ax_shape = axes.shape
            if len(ax_shape) == 2:
                for i in range(ax_shape[0]):
                    for j in range(ax_shape[1]):
                        images[(4*i)+j] = images[(4*i)+j].cpu() if isinstance(images[(4*i)+j], torch.Tensor) else images[(4*i)+j]
                        im = axes[i,j].imshow(images[(4*i)+j])
                        if ax_titles[0] is not None:
                            axes[i,j].set_title(ax_titles[4*i+j])
                        if colorbar:
                            divider = make_axes_locatable(axes[i,j])
                            cax = divider.append_axes('right', size='5%', pad=0.05)
                            fig.colorbar(im, cax=cax, orientation='vertical')
            else:
                for i in range(ax_shape[0]):
                    images[i] = images[i].cpu() if isinstance(images[i], torch.Tensor) else images[i]
                    im = axes[i].imshow(images[i])
                    if ax_titles[0] is not None:
                        axes[i].set_title(ax_titles[i])
                    if colorbar:
                        divider = make_axes_locatable(axes[i])
                        cax = divider.append_axes('right', size='5%', pad=0.05)
                        fig.colorbar(im, cax=cax, orientation='vertical')

        if title is not None:
            plt.title(title)
        
        plt.show()


def crop_image(image: Union[Tensor,np.ndarray], crop_range: list):
    """Crop an image into a range"""

    r_min, r_max, c_min, c_max = crop_range      

    C,*_ = image_info(image)

    if C==1:
        return image[r_min:r_max, c_min:c_max] 
    else:
        if isinstance(image, Tensor):
            return image[:, r_min:r_max, c_min:c_max] 
        elif isinstance(image, np.ndarray):
            return image[r_min:r_max, c_min:c_max, :]
        else:
            raise NotImplementedError('Expected torch.Tensor or np.ndarray') 


def split_mosaic(mosaic: Union[np.ndarray, Tensor], k: list = (2,2)):
    """
        Split a spectral camera mosaic into a multi-channel image
    
        Args:
            mosaic: 
                spectral mosaic, shape: (H,W)
            k: 
                spectral mosaic's kernel sizes, shape: (kx,ky). Default Bayer kernels k=(2,2)
        Return:
            imgs (np.ndarray or torch.Tensor): 
                Images with different wavelengths splitted in channels.
                Shape: (C, H//kx, W//ky) if type == Tensor
                Shape: (H//kx, W//ky, C) if type == np.ndarray

    """

    _, H, W = image_info(mosaic)
    k_row,k_column = k

    x_max = k_row*(H//k_row) # integer number of patterns
    y_max = k_column*(W//k_column)
    
    mosaic = mosaic[0:x_max, 0:y_max]
    all_images = [mosaic[i::k_row, j::k_column] for i in range(k_row) for j in range(k_column)] # some list comprehension will save us here ...

    images = stack_dtype(all_images)

    return images

    
def inv_split_mosaic(imgs: Union[np.ndarray, Tensor], k: list = (2,2)):
    """
        Return multi-channel image to the spectral camera mosaic 
    
        Args:
            imgs (np.ndarray or torch.Tensor): 
                Images with different wavelengths splitted in channels.
                Shape: (C, H//kx, W//ky) if type == Tensor
                Shape: (H//kx, W//ky, C) if type == np.ndarray
            k: 
                spectral mosaic's kernel sizes, shape: (kx,ky). Default Bayer kernels k=(2,2)
        Return:
            mosaic: 
                spectral mosaic, shape: (H,W)

    """

    C, H, W = image_info(imgs)
    k_row,k_column = k
    
    if isinstance(imgs, Tensor):
        mosaic = torch.zeros((H*k_row, W*k_column))
        imgs = [imgs[c] for c in range(C)]
    elif isinstance(imgs, np.ndarray):
        mosaic = np.zeros((H*k_row, W*k_column))
        imgs = [imgs[:,:,c] for c in range(C)]
    else:
        raise NotImplementedError('Expected torch.Tensor or np.ndarray')   

    c = 0
    for i in range(k_row):
        for j in range(k_column):
            mosaic[i::k_row, j::k_column] = imgs[c] 
            c+=1

    return mosaic


def electron_repr(x, background=99.6, gain=2.2):
    return (x - background)/ gain


def digital_repr(x, background=99.6, gain=2.2):
    return (x * gain + background)


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


def downsample(image, downsample_size):
    if image.ndim == 2:
        image = image[None]
    assert image.ndim == 3, f'Image shape should be (C,H,W) instead of {image.shape}'
    patches = image2patches(image, patch_size= [downsample_size,downsample_size], view_as_patches = False)
    image = patches2image(patches.mean(-1).mean(-1)[None,None])[None]
    return image[0]


def upsample_clone(image):
    image = image[None].repeat(4,1,1)
    image = inv_split_mosaic(image)
    return image


def upsample_interpolation(img):
    return interpolate(img[None,None], size=(2,2))[0,0]


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


def set_dataloaders(subsets:  Union[list, np.ndarray, torch.Tensor], batch_size: int = 32):
    
    if not isinstance(subsets, list):
        train_loader = DataLoader(subsets, batch_size=batch_size, shuffle=True)
        return train_loader
    else:
        train_loader = DataLoader(subsets[0], batch_size=batch_size, shuffle=True)
        if len(subsets) == 2:
            valid_loader = DataLoader(subsets[1], batch_size=batch_size, shuffle=False)
            return train_loader, valid_loader
        if len(subsets) == 3:
            test_loader = DataLoader(subsets[2], batch_size=batch_size, shuffle=False)
            return train_loader, valid_loader, test_loader


def poolingOverlap(mat, f, stride=None, method='max', pad=False,
                   return_max_pos=False):
    '''Overlapping pooling on 2D or 3D data.
    Args:
        mat (ndarray): input array to do pooling on the first 2 dimensions.
        f (int): pooling kernel size.
    Keyword Args:
        stride (int or None): stride in row/column. If None, same as <f>,
            i.e. non-overlapping pooling.
        method (str): 'max for max-pooling,
                      'mean' for average-pooling.
        pad (bool): pad <mat> or not. If true, pad <mat> at the end in
               y-axis with (f-n%f) number of nans, if not evenly divisible,
               similar for the x-axis.
        return_max_pos (bool): whether to return an array recording the locations
            of the maxima if <method>=='max'. This could be used to back-propagate
            the errors in a network.
    Returns:
        result (ndarray): pooled array.
    See also unpooling().
    '''
    m, n = mat.shape[:2]
    if stride is None:
        stride = f
    _ceil = lambda x, y: x//y + 1
    if pad:
        ny = _ceil(m, stride)
        nx = _ceil(n, stride)
        size = ((ny-1)*stride+f, (nx-1)*stride+f) + mat.shape[2:]
        mat_pad = np.full(size, 0)
        mat_pad[:m, :n, ...] = mat
    else:
        mat_pad = mat[:(m-f)//stride*stride+f, :(n-f)//stride*stride+f, ...]
    view = asStride(mat_pad, (f, f), stride)
    if method == 'max':
        result = np.nanmax(view, axis=(2, 3), keepdims=return_max_pos)
    else:
        result = np.nanmean(view, axis=(2, 3), keepdims=return_max_pos)
    if return_max_pos:
        pos = np.where(result == view, 1, 0)
        result = np.squeeze(result)
        return result, pos
    else:
        return result

    
def asStride(arr, sub_shape, stride):
    '''Get a strided sub-matrices view of an ndarray.
    Args:
        arr (ndarray): input array of rank 2 or 3, with shape (m1, n1) or (m1, n1, c).
        sub_shape (tuple): window size: (m2, n2).
        stride (int): stride of windows in both y- and x- dimensions.
    Returns:
        subs (view): strided window view.
    See also skimage.util.shape.view_as_windows()
    '''
    s0, s1 = arr.strides[:2]
    m1, n1 = arr.shape[:2]
    m2, n2 = sub_shape[:2]
    view_shape = (1+(m1-m2)//stride, 1+(n1-n2)//stride, m2, n2)+arr.shape[2:]
    strides = (stride*s0, stride*s1, s0, s1)+arr.strides[2:]
    subs = np.lib.stride_tricks.as_strided(
        arr, view_shape, strides=strides, writeable=False)
    return subs