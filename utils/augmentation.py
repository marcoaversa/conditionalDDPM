from email.errors import MultipartConversionError
import random
import wave
from matplotlib.pyplot import jet
import numpy as np
import scipy

import torch
from torch import Tensor
import torchvision.transforms as T

import poppy
from astropy import units as un

from utils.base import stack_dtype, convolve_dtype, image_info,\
                         crop_image, image2patches, patches2image

from utils.base import Timer

from typing import Union

import pyjetraw4ai_proto as jetraw4ai


def set_global_seed(seed):
    torch.random.manual_seed(seed)
    np.random.seed(seed % (2**32 - 1))
    random.seed(seed)


class ComposeState(T.Compose):
    def __init__(self, transforms):
        self.transforms = []
        self.mask_transforms = []

        for t in transforms:
            self.transforms.append(t)

        self.seed = None
        self.retain_state = False

    def __call__(self, x):
        if self.seed is not None:   # retain previous state
            set_global_seed(self.seed)
        if self.retain_state:    # save state for next call
            self.seed = self.seed or torch.seed()
            set_global_seed(self.seed)
        else:
            self.seed = None    # reset / ignore state
            
        if isinstance(x, (list, tuple)):
            return self.apply_sequence(x)
        else:
            return self.apply_img(x)

    def apply_img(self, img):
        for t in self.transforms:
            img = t(img)
        return img
    
    def apply_sequence(self, seq):
        self.retain_state=True
        seq = list(map(self, seq))
        self.retain_state=False
        return seq

class ImageDegradation():       
    """
        Downgrade the image resolution metrologically

        Args:
            pixel_size: 
                sensor's pixel size (in m)
            wavelength: 
                wavelength (in m). 
                Example: [[lambda_1,lambda_2,..],[lambda_i, lambda_(i+1), .. ]]
                Default: [[R,G],[G,B]]
            height:
                orbital height (in m)
            radius:
                mirror radius, (in m)
            PSF_FOV:
                field of view of the PSF, choose an odd number, so that the PSF is centered
            source_GSD:
                the GSD of the source images, in m (1 pixel)
            emulated_GSD:
                the emulated GSD for the satellite (1 pixel)
            emulated_exposure_time:
                the emulated exposure time of the satellite (s)
            angle: 
                motion blur direction, degrees range=[0,180]
            device: 
                choose device, e.g.: [cpu,cuda:0]
    """

    def __init__(
                self,
                emulated_pixel_size: float =  2.74e-6, 
                emulated_focal_length: float = 1.644,
                emulated_height: float = 600e3, 
                emulated_radius: float = 0.2, 
                emulated_exposure_time: float = 2e-4,
                source_pixel_size: float =  2.43e-6, 
                source_focal_length: float = 1.21e-2,
                source_height: float = 250, 
                source_exposure_time: float = 1/15,
                source_gain: Union[list,float] = [8.450,8.445,8.427,8.382], 
                source_black_level: Union[list,float] = [4099.3, 4101.0, 4100.4, 4099.8], 
                source_readout_noise: Union[list,float] = [22.6,22.7,22.4,22.9], 
                source_white_level: Union[list,float] = [65535,65535,65535,65535],
                PSF_FOV: float = 5.,
                wavelengths: Union[float,list] = [0.63e-6, 0.532e-6, 0.532e-6, 0.465e-6],
                pattern: list = [2,2],  
                angle: float = 0, # [0,180]
                device: str = None):

        assert 1e-6 <= emulated_pixel_size <= 11e-6,        'Pixel size can be only in the range [1e-6,11e-6]'
        assert 0.1 <= emulated_focal_length <= 3.0,         'Focal length can be only in the range [0.1,3.0]'
        assert 0.05 <= emulated_radius <= 0.5,              'Mirror radius can be only in the range [0.05,0.5]'
        assert 1e-6 <= emulated_exposure_time <= 1e-3,     'Exposure time can be only in the range [1e-7,1e-5]'

        self.emulated_pixel_size = emulated_pixel_size #* un.meter/un.pixel
        self.emulated_focal_length = emulated_focal_length #* un.meter
        self.emulated_height = emulated_height #* un.meter
        self.emulated_radius = emulated_radius #* un.meter
        self.emulated_exposure_time = emulated_exposure_time #* un.second
        self.emulated_F = emulated_focal_length/(2*emulated_radius) # F number
        self.emulated_GSD = self.gsd(emulated_height, emulated_pixel_size, emulated_focal_length)

        self.wavelengths = wavelengths if isinstance(wavelengths, list) else [wavelengths,]
        self.PSF_FOV = PSF_FOV

        self.source_pixel_size = source_pixel_size #* un.meter/un.pixel
        self.source_focal_length = source_focal_length #*un.meter
        self.source_height = source_height #*un.meter
        self.source_exposure_time = source_exposure_time #* un.second
        self.source_F = 8.0
        self.source_GSD = self.gsd(source_height, source_pixel_size, source_focal_length)
        self.params = [source_gain, source_black_level, source_readout_noise, source_white_level]

        self.Idrop() # Check if parameters are suitable for the degradation

        self.is_mosaic = True

        # number of wavelengths in the mosaic pattern (n_row, n_column). Default Bayer with the common RGGB pattern.
        # for a grey image the pattern should be [1,1]
        self.pattern = pattern

        # this is used to resample individual channels, should be integer
        # ideally, it should be odd, so that the oversampled psf is centered on a pixel
        self.even_sampling_ratio()
        
        # compute psf for each wavelength
        self.psf_kernels = self.psf()
        self.mb_kernels = [self.rotate_kernel(self.motion_blur(),angle) for w in self.wavelengths]

        if device:
            self.device = device
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def gsd(self, height, pixel_size, focal_length):
        #https://www.propelleraero.com/blog/ground-sample-distance-gsd-calculate-drone-data/#:~:text=What%20is%20ground%20sample%20distance%20or%20GSD%3F,one%20pixel%20on%20the%20ground.
        return height * pixel_size / focal_length

    def Idrop(self):
        # with source_exposure_time=1/15 , self.emulated_exposure_time=2e-4, radius=0.2 --> minimum focal length is 0.438
        Idrop_fnumber = (self.source_F/self.emulated_F)**2
        Idrop_pixel_size = (self.emulated_pixel_size/self.source_pixel_size)**2
        Idrop_exposure_time = self.emulated_exposure_time/self.source_exposure_time
        Idrop = Idrop_fnumber*Idrop_exposure_time*Idrop_pixel_size
        assert Idrop <= 1, self.check_parameters(Idrop_fnumber, Idrop_exposure_time, Idrop_pixel_size)
        return Idrop

    def check_parameters(self, Idrop_fnumber: float, Idrop_exposure_time: float, Idrop_pixel_size: float):
        
        print('Emulated intensity should be lower or equal to the source one.') 
        print('Change the emulated focal_length, exposure time, pixel size or radius.')
        print('Fixed others parameters, you should choose:')

        ind = np.argmax([Idrop_fnumber, Idrop_exposure_time, Idrop_pixel_size]) 
        if ind == 0:
            print(f'a) A focal length higher than {self.min_focal_length()}') 
            print(f'b) A mirror radius lower than {self.max_radius()}') 
        elif ind == 1:
            print(f'a) A time exposure lower than {self.max_time_exposure()}') 
        elif ind == 2:
            print(f'a) A pixel size lower than {self.max_pixel_size()}') 

    def even_sampling_ratio(self):
        old_focal_length = self.emulated_focal_length
        sampling_ratio = self.emulated_GSD/self.source_GSD
        self.sampling_ratio = [round(a/b) for a,b in zip([sampling_ratio,sampling_ratio],self.pattern)]
        while self.sampling_ratio[0]%self.pattern[0] != 0. or self.sampling_ratio[1]%self.pattern[1] != 0.:
            self.emulated_focal_length -= 0.001
            self.emulated_GSD = self.gsd(self.emulated_height, self.emulated_pixel_size, self.emulated_focal_length)
            sampling_ratio = self.emulated_GSD/self.source_GSD
            self.sampling_ratio = [round(a/b) for a,b in zip([sampling_ratio,sampling_ratio],self.pattern)]
        if old_focal_length != self.emulated_focal_length:
            print(f'\nFocal length decreased from {old_focal_length:.3f} to {self.emulated_focal_length:.3f} to obtain a downsampling ratio ({self.sampling_ratio[0]},{self.sampling_ratio[1]}) which is a multiplier of the CFA ({self.pattern[0]},{self.pattern[1]}).')  
        if self.sampling_ratio[0] > 40 or self.sampling_ratio[1] > 40:  
            print(f'\nDownsampling ratio (d_rows,d_columns) = ({self.sampling_ratio[0]},{self.sampling_ratio[1]}) is bigger than (40,40) \nThe code can run slowly. \nReduce the emulated focal length or increase the mirror radius for a faster data generation.')        
        # assert self.sampling_ratio[0] <= 40 or self.sampling_ratio[1] <= 40, f'Downsampling ratio ({self.sampling_ratio[0]},{self.sampling_ratio[1]}) cannot be bigger than 40, reduce the emulated focal length or increase the mirror radius'
        # assert self.sampling_ratio[0]%self.pattern[0] == 0. or self.sampling_ratio[1]%self.pattern[1] == 0. , 'Downsampling ratio should be a multiplier of the CFA, change the emulated focal length or the mirror radius'

        
    def min_focal_length(self):
        "Fixed others emulated parameters, it returns the minimum possible focal length that it's possible to simulate"
        Idrop_pixel_size = (self.emulated_pixel_size/self.source_pixel_size)**2
        Idrop_exposure_time = self.emulated_exposure_time/self.source_exposure_time
        Idrop_fnumber = 1/(Idrop_pixel_size*Idrop_exposure_time)
        emulated_F = self.source_F / np.sqrt(Idrop_fnumber)         
        min_focal = emulated_F*2*self.emulated_radius
        return min_focal

    def max_radius(self):
        "Fixed others emulated parameters, it returns the maximum possible radius that it's possible to simulate"
        Idrop_pixel_size = (self.emulated_pixel_size/self.source_pixel_size)**2
        Idrop_exposure_time = self.emulated_exposure_time/self.source_exposure_time
        emulated_F = self.source_F / np.sqrt(1/(Idrop_pixel_size*Idrop_exposure_time))         
        max_radius =  self.emulated_focal_length/(2*emulated_F)
        return max_radius

    def max_pixel_size(self):
        "Fixed others emulated parameters, it returns the maximum possible pixel size that it's possible to simulate"
        Idrop_exposure_time = self.emulated_exposure_time/self.source_exposure_time
        Idrop_fnumber = (self.source_F/self.emulated_F)**2
        Idrop_pixel_size = 1/(Idrop_fnumber*Idrop_exposure_time)
        print(Idrop_pixel_size)
        max_pixel_size = self.source_pixel_size * np.sqrt(Idrop_pixel_size)  
        return max_pixel_size    

    def max_time_exposure(self):
        "Fixed others emulated parameters, it returns the maximum possible exposure time that it's possible to simulate"
        Idrop_pixel_size = (self.emulated_pixel_size/self.source_pixel_size)**2
        Idrop_fnumber = (self.source_F/self.emulated_F)**2
        Idrop_exposure_time = 1/(Idrop_pixel_size*Idrop_fnumber)    
        max_exposure_time = self.source_exposure_time * Idrop_exposure_time
        return max_exposure_time

    def psf(self):     
        """
            Estimate the point spread function (PSF) respect to the simulated satellite design parameters

            Return:
                kernels (list of np.ndarray): psf kernels
        """
        kernels = []
        for i ,w in enumerate(self.wavelengths):
            # for a Bayer sensor, please choose a source and emulated GSD such that the sampling ratio is an even number.
            pixel_angle = ((self.emulated_GSD/self.emulated_height)*un.radian/un.pixel).to(un.arcsec/un.pixel) # small angle.
            sampling_ratio = [s if s%2 != 0 else s+1 for s in self.sampling_ratio]
            osys = poppy.OpticalSystem(oversample=sampling_ratio[0] if i<self.pattern[0] else sampling_ratio[1])
            osys.add_pupil( poppy.CircularAperture(radius=self.emulated_radius))    # pupil radius in meters
            osys.add_detector(pixelscale=pixel_angle, fov_pixels=self.PSF_FOV)  # image plane coordinates in arcseconds
            psf = osys.calc_psf(w, normalize="last")  # normalize the kernel within itself.
            kernels.append(psf[0].data) 
        return kernels

    def vignetting(self, mosaic: Tensor, factor: float = 1.):
        """
            Add vignetting to the image   

            Args:
                mosaic: mosaic image or grey image
                factor: amount of vignetting applied. Value between [0,1]
            Return:
                mosaic: mosaic image or grey image with vignetting applied
        """
        #https://stackoverflow.com/questions/62045155/how-to-create-a-transparent-radial-gradient-with-python
        #https://link.springer.com/article/10.1007/s11760-016-0941-2#Sec3

        f=10**(2*(1-factor))

        if self.is_mosaic:
            image = self.split_mosaic(mosaic.squeeze(), self.pattern) if self.pattern else mosaic.squeeze()
            C, *_ = image_info(image)
            for c in C:
                image[c] = self.vignetting(image[c],factor)
            mosaic = self.inv_split_mosaic(image, self.pattern) if self.pattern else image
        
        else:
            _, H, W = image_info(mosaic)
            Y = torch.linspace(-1, 1, H)[None, :]
            X = torch.linspace(-1, 1, W)[:, None]
            Y = Y[:,int(((W*f)//2)-W//2):int(((W*f)//2)+W//2)]
            X = X[int(((H*f)//2)-H//2):int(((H*f)//2)+H//2)]
            # Create radial alpha/transparency layer. 1 in centre, 0 at edge
            alpha = torch.sqrt(X**2 + Y**2)
            alpha = 1 - torch.clip(alpha,0,1)
            mosaic = mosaic*alpha

        return mosaic

    def motion_blur(self):
        """
            Estimate the motion blur (MB) kernel respect to the simulated time exposure

            Return:
                kernel (np.ndarray): motion blur kernel
        """
        
    #     #https://physics.stackexchange.com/questions/313422/why-doesnt-earth-appear-smudgy-or-blurred-in-space-photographs-due-to-its-fast

        mu = 3.986e14 # Standard gravitational parameter (m**3 / s**2)
        R_T = 6.371e6 # Earth radius (m)

        # Orbital Period Satellite
        # https://en.wikipedia.org/wiki/Circular_orbit

        T = 2*np.pi * np.sqrt( (R_T+self.emulated_height)**3 / (mu) )
        t = T / (2*np.pi*R_T) # Within that time 1 pixel corresponds to 1 meter on Earth
        # If the exposure time of the satellite sensor is higher that this t, the camera is affected by motion blur

        ratio = self.emulated_exposure_time/t
        ratio_row, _ = ratio*self.sampling_ratio[0], ratio*self.sampling_ratio[1]

        mb_len = int(ratio_row) 
        subpixel_mb = ratio_row%int(ratio_row)
        mb_len += 1 if subpixel_mb != 0.0 else 0 
        size = mb_len if mb_len%2 != 0 else mb_len +1

        kernel = np.zeros((size,size))
        kernel[int(size//2)] = np.ones(size)
        kernel[int(size//2),-1] = 1 if size == mb_len else 0
        if subpixel_mb != 0.0:
            kernel[int(mb_len//2),0] = subpixel_mb/2
            kernel[int(mb_len//2),-1 if size == mb_len else -2] = subpixel_mb/2

        kernel /= kernel.sum()

        return kernel


    def convolve_mosaic(self,mosaic: Union[Tensor, np.ndarray], kernels: Union[Tensor,np.ndarray]):
        """
            Split a raw image mosaic into channels and convolve kernels channel-wise

            Args:
                - mosaic: original raw image
                - kernels: kernels for each wavelength of the raw image, len(kernels) == len(wavelengths)
            Return:
                - mosaic: convolved raw image
        """
        assert self.is_mosaic, 'Expected image in the mosaic shape'
        image = self.split_mosaic(mosaic.squeeze(), self.pattern) if self.pattern else mosaic.squeeze()
        image = self.convolve(image, kernels).squeeze()
        mosaic = self.inv_split_mosaic(image, self.pattern) if self.pattern else image
        return mosaic[None].to(self.device) if isinstance(mosaic, Tensor) else mosaic


    def convolve(self, img: Union[Tensor, np.ndarray], kernels: Union[Tensor,np.ndarray]):
        """
            Convolve an image with different channels with kernels

            Args:
                - img: image, shape = (B,C,H,W)
                - kernels: kernels for each wavelength of the raw image, len(kernels) == len(wavelengths)
            Return:
                - img: convolved image
        """
        assert self.is_mosaic == False, 'Image should not be in the mosaic shape'
        kernels = [Tensor(k).to(self.device) for k in kernels] if isinstance(img, Tensor) else kernels
        kernels = stack_dtype(kernels)
        padding=False
        img = convolve_dtype(img,kernels,padding)
        return img[None].to(self.device) if isinstance(img, Tensor) else img


    def split_mosaic(self, mosaic: Union[np.ndarray, Tensor], k: list = (2,2)):
        """
            Split a spectral camera mosaic into a multi-channel image
        
            Args:
                mosaic: spectral mosaic, shape: (H,W)
                k: spectral mosaic's kernel sizes, shape: (kx,ky). Default Bayer kernels k=(2,2)
            Return:
                imgs: 
                    Images with different wavelengths splitted in channels.
                    Shape: (C, H//kx, W//ky) if type == Tensor
                    Shape: (H//kx, W//ky, C) if type == np.ndarray

        """

        assert self.is_mosaic, 'Raw image should be a mosaic to be splitted in channels'
        assert mosaic.ndim == 2, f'Image shape should be (H,W) instead of {mosaic.shape}'
        _, H, W = image_info(mosaic)
        k_row,k_column = k

        x_max = k_row*(H//k_row) # integer number of patterns
        y_max = k_column*(W//k_column)
        
        mosaic = mosaic[0:x_max, 0:y_max]
        all_images = [mosaic[i::k_row, j::k_column] for i in range(k_row) for j in range(k_column)] # some list comprehension will save us here ...

        images = stack_dtype(all_images)

        self.is_mosaic = False

        return images

    
    def inv_split_mosaic(self, imgs: Union[np.ndarray, Tensor], k: list = (2,2)):
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

        assert self.is_mosaic == False, 'Raw image should be splitted in channels to be inverted into a mosaic'
        assert imgs.ndim == 3, f'Image shape should be (C,H,W) if torch.Tensor or (H,W,C) if np.ndarray instead of {imgs.shape}'
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
                
        self.is_mosaic = True

        return mosaic

    def downscale_raw(self, mosaic: Union[Tensor, np.ndarray]):
        """
            Downscale a raw image by a factor scale=(s_row,s_column). 

            Args:
                mosaic: Raw image to downscale, shape = (H, W)
            Return:
                mosaic_downscaled: Raw image downscaled
        """
        
        mosaic = mosaic.squeeze()
        assert mosaic.ndim == 2, f'Expected image in the shape = (Height, Weight), Image used in the shape = {mosaic.shape}'
        _, H, W = image_info(mosaic)
    
        s_row, s_column = self.sampling_ratio[0], self.sampling_ratio[1]
        C_row, C_column = self.pattern

        # Crop the raw image by a multiplier of the sampling_ratio*number_wavelengths_per_axis  
        # r_crop_range = (H//(s_row*C_row))*(s_row*C_row)
        # c_crop_range = (W//(s_column*C_column))*(s_column*C_column)
        # Crop the raw image by a multiplier of the sampling_ratio (sampling_ratio is a multiplier of the CFA)
        r_crop_range = (H//(s_row))*(s_row)
        c_crop_range = (W//(s_column))*(s_column)
        mosaic = crop_image(mosaic, [0,r_crop_range,0,c_crop_range])

        # Tile image into patches
        patches = image2patches(mosaic,(s_row,s_column))
        n_h,n_w, *_ = patches.shape

        # Initialize matrices with the downsampled size
        if isinstance(mosaic, Tensor):
            raw_d = torch.zeros((n_h,n_w)).to(self.device)
            ref_pattern = torch.zeros_like(mosaic)
        elif isinstance(mosaic, np.ndarray):
            raw_d = np.zeros((n_h,n_w))    
            ref_pattern = np.zeros_like(mosaic)           
        else:
            raise NotImplementedError("Expected torch.Tensor or np.ndarray")

        # Create a map with wavelengths positions
        for k in range(C_row):
            for l in range(C_column):
                ref_pattern[k::C_row, l::C_column] = k*C_row + l
        ref_patches = image2patches(ref_pattern,(s_row,s_column))

        # For each block we take the mean of the pixels with the same wavelength
        for k in range(C_row):
            for l in range(C_column):
                blocks = patches[k::C_row, l::C_column]
                # ref_blocks = ref_patches[k::C_row, l::C_column].clone() if isinstance(ref_patches, Tensor) else ref_patches[k::C_row, l::C_column].copy()
                ref_blocks = ref_patches[k::C_row, l::C_column]
                ref_blocks = ref_blocks == (k*C_row + l)

                raw_d[k::C_row, l::C_column] = (blocks*ref_blocks).sum(-1).sum(-1) / ref_blocks.sum(-1).sum(-1)
        
        # Map back from patches to image
        mosaic_downscaled = patches2image(raw_d.reshape(n_h, n_w,1,1))

        return mosaic_downscaled[None] if isinstance(mosaic_downscaled, Tensor) else mosaic_downscaled

    def diffusion(
                self,
                img: Union[torch.Tensor, np.ndarray],
                factor: float,
                G: float, 
                B: float, 
                N: float, 
                W: float):

        params = jetraw4ai.Parameters(G, B, N, W)
        jetraw_im = jetraw4ai.JetrawImage(img, params)
        
        el_repr = jetraw_im.electron_repr()
        v_max = el_repr.image_data.max()
        v_max_el = int((W-B)/G)
        f = int(factor*(v_max_el-v_max))
        el_repr.image_data += f
        diffused_img = el_repr.replace_noise().digital_repr().image_data
        ratio = diffused_img.mean()/img.mean()
        diffused_img /= ratio
        return diffused_img

    def add_clouds(
                self,
                img: Union[torch.Tensor, np.ndarray],
                factor: float):

        assert factor < 1. or factor >= 0., 'Intensity scale factor should be in the range [0.,1.)'
        assert img.ndim == 2, 'Input image should be in the raw pattern format (H,W)'
        if isinstance(img, torch.Tensor):
            img = np.array(img.cpu()).astype(np.float64)

        G,B,N,W = self.params

        if self.is_mosaic:
            img = self.split_mosaic(img, self.pattern)
            for i in range(self.pattern[0]):
                for j in range(self.pattern[1]):
                    index = i*self.pattern[0]+j
                    img[:,:,index] = self.diffusion(img[:,:,index], factor, G[index], B[index], N[index], W[index])
            img = self.inv_split_mosaic(img, self.pattern)
        else: # grey sensor
            self.diffusion(img, factor, G, B, N, W)
                    
        return torch.Tensor(img).to(self.device)                 

    def electron_repr(self, image_data, G, B):
        return (image_data - B)/ G

    def digital_repr(self, image_data, G, B):
        return image_data * G + B

    def scale_exposure(
                    self,
                    image_data: torch.Tensor, 
                    factor: float, 
                    G: float, 
                    B: float, 
                    N: float, 
                    W: float):

        if not 0 <= factor <= 1:
            raise ValueError("factor must be between 0 and 1")
            
        if factor != 1:
            image_data = self.electron_repr(image_data, G, B)
            scaled_data = factor * image_data
            noise_var = (1 - factor) * torch.clip(scaled_data, 0, None) + (1 - factor ** 2) * (N / G) ** 2
            scaled_data = scaled_data + torch.normal(0, torch.sqrt(noise_var))
            scaled_data = self.digital_repr(scaled_data, G, B)
            return torch.clip(scaled_data,0,W)
        else:
            return image_data
        

    def decrease_intensity(
                    self,
                    img: torch.Tensor, 
                    factor: float =  1.):
        """
            Decrease the intensity of a raw image by a factor.
            REMARK: In the calibration curve (G == z, B == c, N ==  sigma, W == max_value)

            Args:
                img: raw image
                factor: percentage of intensity rescaling, range (0.,1.]

            Return:
                img_rescaled: Rescaled raw image    
        """

        img = img.squeeze()
        assert factor <= 1. or factor > 0., 'Intensity scale factor should be in the range (0.,1.]'
        assert img.ndim == 2, 'Input image should be in the raw pattern format'
        # if isinstance(img, torch.Tensor):
        #     img = np.array(img.cpu()).astype(np.float64)

        G,B,N,W = self.params

        if factor < 1.:
            if self.is_mosaic:
                img = self.split_mosaic(img, self.pattern)
                for i in range(self.pattern[0]):
                    for j in range(self.pattern[1]):
                        index = i*self.pattern[0]+j
                        img[index] = self.scale_exposure(img[index], factor, G[index], B[index], N[index], W[index])
                img = self.inv_split_mosaic(img, self.pattern)
            else: # grey sensor
                img = self.scale_exposure(img,factor, G, B, N, W)

        return img[None]    

    def replace_noise(
                    self,
                    img: Union[torch.Tensor,np.ndarray]):
        """
            Resample the noise of a raw image.

            Args:
                img: raw image

            Return:
                img: raw image with noise resampled    
        """

        img = img.squeeze()
        assert img.ndim == 2, 'Input image should be in the raw pattern format'
        if isinstance(img, torch.Tensor):
            img = np.array(img.cpu()).astype(np.float64)

        B,G,N,W = self.params

        if self.is_mosaic:
            img = self.split_mosaic(img, self.pattern)
            for i in range(self.pattern[0]):
                for j in range(self.pattern[1]):     
                    index = i*self.pattern[0]+j   
                    params = jetraw4ai.Parameters(G[index], B[index], N[index], W[index])
                    jetraw_im = jetraw4ai.JetrawImage(img[index], params)
                    img[index] = jetraw_im.replace_noise().image_data
            img = self.inv_split_mosaic(img, self.pattern)
        else: # grey sensor
            params = jetraw4ai.Parameters(G, B, N, W)
            jetraw_im = jetraw4ai.JetrawImage(img, params)
            img = jetraw_im.replace_noise().image_data

        return Tensor(img[None]).to(self.device)   

    def rotate_kernel(self,kernel: np.ndarray, angle: float):
        return scipy.ndimage.rotate(kernel, angle)


    def __call__(self, x):
        x = self.convolve_mosaic(x, self.psf_kernels)
        x = self.convolve_mosaic(x, self.mb_kernels)
        if x.max()>1: # check if it's mask
            x = self.decrease_intensity(x, self.Idrop())
        x = self.downscale_raw(x)
        if x.max()>1: # check if it's mask
            x = self.replace_noise(x)
        """TODO: First 10 lines noisy. Fix it"""
        x = x[:,10:] 
        return x
        

    def __repr__(self):
        return self.__class__.__name__


class RandomRotate90():  # Note: not the same as T.RandomRotation(90)
    def __call__(self, x):
        x = x.rot90(random.randint(0, 3), dims=(-1, -2))
        return x

    def __repr__(self):
        return self.__class__.__name__


class AddGaussianNoise():
    def __init__(self, std=0.01):
        self.std = std

    def __call__(self, x):
        # noise = torch.randn_like(x) * self.std
        # out = x + noise
        # debug(x)
        # debug(noise)
        # debug(out)
        return x + torch.randn_like(x) * self.std

    def __repr__(self):
        return self.__class__.__name__ + f'(std={self.std})'