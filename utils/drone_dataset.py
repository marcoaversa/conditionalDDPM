import os
import shutil
from pathlib import Path
from tqdm import tqdm
from PIL import Image

import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split

import random

from .base import check_image_folder_consistency, image2patches, list_images_in_dir, load_image, split_mosaic, inv_split_mosaic, poolingOverlap
from .augmentation import ComposeState, RandomRotate90, ImageDegradation
from .tifffile_dng import dng_from_template
from .resize_right import resize


'''Datasets'''


class DatasetSegmentation(Dataset):
    """Creates a dataset of images in `img_dir` and corresponding masks in `mask_dir`.
    Corresponding mask files need to contain the filename of the image.
    Files are expected to be of the same filetype.

    Args:
        img_dir (str): path to image folder
        mask_dir (str): path to mask folder
        transform (callable, optional): transformation to apply to image and mask
        bits (int, optional): normalize image by dividing by 2^bits - 1
    """

    task = 'segmentation'

    def __init__(self, img_dir, mask_dir, is_mosaic=False, is_mask=True, transform=None):

        self.img_dir = img_dir
        self.mask_dir = mask_dir

        self.images = list_images_in_dir(img_dir)
        self.masks = list_images_in_dir(mask_dir)

        check_image_folder_consistency(self.images, self.masks)

        self.is_mosaic = is_mosaic
        self.is_mask = is_mask
        self.transform = transform

    def __repr__(self):
        rep = f"{type(self).__name__}: ImageFolderDatasetSegmentation[{len(self.images)}]"
        for n, (img, mask) in enumerate(zip(self.images, self.masks)):
            rep += f'\nimage: {img}\tmask: {mask}'
            if n > 10:
                rep += '\n...'
                break
        return rep

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        img = load_image(self.images[idx])
        mask = load_image(self.masks[idx])
        
        if self.is_mosaic:
            img = split_mosaic(img)
            if self.is_mask:
                mask = poolingOverlap(mask, 2, method='mean')
            else:
                mask = split_mosaic(mask)
            
        # Apply threshold
        if self.is_mask:
            mask = (mask > 0.2).astype(np.float32)

        if self.transform is not None:
            img,mask = self.transform((img,mask))
        return img, mask


class DroneDataset(DatasetSegmentation):
    """Dataset consisting of full-sized numpy images and masks. Images are normalized to range [0, 1].
    """

    def __init__(self, transform=None, is_mosaic=False, data_dir='./data/drone/original'):
        img_dir = os.path.join(data_dir, 'images')
        mask_dir = os.path.join(data_dir, 'masks')
        super().__init__(img_dir=img_dir, mask_dir=mask_dir, is_mosaic=is_mosaic, is_mask=True, transform=transform)


class DroneDatasetTiled(DatasetSegmentation):
    """Dataset consisting of tiled numpy images and masks. Images are in range [0, 1]
    Args:
        tile_size (int, optional): size of the tiled images.
    """

    def __init__(self, tile_size: int = 128, is_mosaic=False, transform = None, data_dir='data/drone/original'):

        if is_mosaic:
            tile_size_name = tile_size
            tile_size = tile_size*2
        else:
            tile_size_name = tile_size
            
        img_dir = os.path.join(data_dir, f'images_tiles_{tile_size_name}')
        mask_dir = os.path.join(data_dir, f'masks_tiles_{tile_size_name}')

        if not os.path.exists(img_dir) or not os.path.exists(mask_dir):
            dataset_full = DroneDataset()
            print("tiling dataset..")
            create_tiles_segmentation(dataset_full, img_dir, mask_dir, tile_size=tile_size)

        super().__init__(img_dir=img_dir, mask_dir=mask_dir, is_mosaic=is_mosaic, is_mask=True, transform=transform)


class DroneDatasetEmulated(DatasetSegmentation):
    def __init__(self, f=1.644, r=0.1, t=180e-6, tile_size: int = 128, is_mosaic=False, transform = None, force_tiling=False, data_dir='data/drone/emulated'):

        if is_mosaic:
            tile_size_name = tile_size
            tile_size = tile_size*2

        img_dir = os.path.join(data_dir, f'images_tiles_{tile_size_name}')
        mask_dir = os.path.join(data_dir, f'masks_tiles_{tile_size_name}')

        if not os.path.exists(img_dir) or not os.path.exists(mask_dir) or force_tiling:
            dataset_full = DroneDataset(is_mosaic=False, img_dir=os.path.join(data_dir, 'images'), mask_dir=os.path.join(data_dir, 'masks'))
            params = f'f{f:.3f}_d{r:.2f}_t{t:.5f}'.replace('.','_')
            dataset_full.images = list(filter(None, [img if params in img else None for img in dataset_full.images]))
            dataset_full.masks = list(filter(None, [mask if params in mask else None for mask in dataset_full.masks]))
            print("tiling dataset..")
            create_tiles_segmentation(dataset_full, img_dir, mask_dir, tile_size=tile_size)

        super().__init__(img_dir=img_dir, mask_dir=mask_dir, is_mosaic=is_mosaic, is_mask=True, transform=transform)
          
          
class DroneUpsampleTiled(DatasetSegmentation):
    def __init__(
                self, 
                f=1.644, 
                r=0.1, 
                t=180e-6, 
                height=600e3,
                tile_size: int = 128, 
                original_path = 'data/drone/original',
                emulated_path = 'data/drone/emulated',
                is_mosaic=True, 
                transform = None, 
                force_tiling=False):
            
        # threshold=0.2

        degradation = ImageDegradation(
            emulated_focal_length = f,
            emulated_radius = r,
            emulated_exposure_time = t)
        
        f=degradation.emulated_focal_length

        img_dir = os.path.join(original_path, f'images_tiles_{tile_size}')
        mask_dir = os.path.join(emulated_path, f'images_tiles_{tile_size}')

        if not os.path.exists(img_dir) or not os.path.exists(mask_dir) or force_tiling:            
            create_emulated_images(f=f, r=r, t=t, height=height, in_path=original_path, out_path=emulated_path, device='cuda:0', force_generation=True)
            if force_tiling and os.path.exists(os.path.join(original_path, f'images_tiles_{tile_size}')):
                shutil.rmtree(os.path.join(original_path, f'images_tiles_{tile_size}'))
                shutil.rmtree(os.path.join(original_path, f'masks_tiles_{tile_size}'))
            
            dataset = DroneDataset(data_dir=original_path)
            x = torch.Tensor(dataset[0][0])[None,None].cuda()
            
            y = degradation.convolve_mosaic(x, degradation.psf_kernels)
            y = degradation.convolve_mosaic(y, degradation.mb_kernels)
            
            y_crop = (x.squeeze().shape[0]-y.squeeze().shape[0])//2
            x_crop = (x.squeeze().shape[1]-y.squeeze().shape[1])//2
            
            G,B,*_ = degradation.params
            G,B = list(map(torch.tensor, (G,B)))
            
            # During the emulation, the image is convolved with several kernels which reduce the size.
            # Here we crop the image such that we have the same region
            # The intensity is rescaled to be compatible with the emulated one
            # Remark: We are not resampling the noise to don't lose information
            crop = ComposeState([ 
                                 T.Lambda(lambda img: img[y_crop:-y_crop,x_crop:-x_crop]),
                                 T.Lambda(lambda x: split_mosaic(x)),
                                 T.ToTensor(),
                                 T.Lambda(lambda x: x.to('cuda:0')),
                                 T.Normalize(B,G),
                                 T.Lambda(lambda img: img*degradation.Idrop()),
                                 T.Normalize(torch.zeros_like(B),1/G),
                                 T.Normalize(-B,torch.ones_like(G)),
                                 T.Lambda(lambda x: np.array(x.permute(1,2,0).squeeze().cpu())),
                                 T.Lambda(lambda x: inv_split_mosaic(x)),
                                 ])

            dataset_ref = DroneDataset(data_dir=original_path, transform=crop)
            print('Tiling original images\n')
            create_tiles_segmentation(dataset_ref, Path(original_path, f'images_tiles_{tile_size}'), Path(original_path, f'masks_tiles_{tile_size}'), tile_size=tile_size*2)
            
            c,d = torch.Tensor(dataset_ref[0][0]).shape[0], torch.Tensor(dataset_ref[0][0]).shape[1]
            
            resample = ComposeState([
                T.Lambda(lambda x: split_mosaic(x)),
                T.ToTensor(),
                T.Lambda(lambda x: x.to('cuda:0')),
                T.Lambda(lambda x: resize(x,out_shape=(4,c//2,d//2))),
                T.Lambda(lambda x: np.array(x.permute(1,2,0).squeeze().cpu())),
                T.Lambda(lambda x: inv_split_mosaic(x)),
                # T.Lambda(lambda x: (x > threshold) if x.max() < 10 else x),
                ])
            
            dataset = DroneDataset(data_dir=emulated_path, transform=resample)
            print('Tiling emulated images\n')
            create_tiles_segmentation(dataset, Path(emulated_path, f'images_tiles_{tile_size}'), Path(emulated_path, f'masks_tiles_{tile_size}'), dataset_ref=dataset_ref, tile_size=tile_size*2)

        super().__init__(img_dir=img_dir, mask_dir=mask_dir, is_mosaic=is_mosaic, is_mask=False, transform=transform)


'''Tools'''


def set_global_seed(seed=None):
    seed = 6662177862330650406 if seed is None else seed
    torch.random.manual_seed(seed)
    np.random.seed(seed % (2**32 - 1))
    random.seed(seed)
    
    
def get_mu_sigma(dataloader, is_mosaic=False, dim=0, device='cuda'):
    '''
        Compute Mean and STD for a dataloader composed by images.
        Return:
            mu: Mean. 1d torch.Tensor, the length is the number of channels for the image
            sigma: STD. 1d torch.Tensor, the length is the number of channels for the image
    
    '''
    
    mu = torch.zeros(4).to(device)
    sigma = torch.zeros(4).to(device)
    for x,y in dataloader:
        if dim == 0:
            var = x
        elif dim == 1:
            var = y
        else:
            raise Exception("Select between dim=[0,1] (corresponds to x,y in the dataset)") 
        if is_mosaic:
            var = split_mosaic(var)[None]
        mu += var.sum(0).sum(-1).sum(-1)/(var.shape[0]*var.shape[-1]*var.shape[-2])
    mu = mu/len(dataloader)
    for x,y in dataloader:
        if dim == 0:
            var = x
        elif dim == 1:
            var = y
        else:
            raise Exception("Select between dim=[0,1] (corresponds to x,y in the dataset)") 
        if is_mosaic:
            var = split_mosaic(x)[None]
        sigma += ((var-torch.einsum('ijkl, j -> ijkl', torch.ones_like(var), mu)).sum(0).sum(-1).sum(-1)/(var.shape[0]*var.shape[-1]*x.shape[-2]))**2
    sigma = torch.sqrt(sigma/len(dataloader))
    return mu, sigma


def get_dataloader(
    f: float = 1.644, 
    r: float = 0.5, 
    t: float = 180e-6, 
    height: float =  600e3,
    transform = None, 
    tile_size: int = 128, 
    batch_size: int = 8, 
    force_tiling: bool = False, 
    dataset_type: str = 'emulated'):
    
    set_global_seed()
    
    if dataset_type == 'emulated':
        dataset = DroneDatasetEmulated(f=f, r=r, t=t, tile_size=tile_size, is_mosaic=True, transform=transform, force_tiling=force_tiling)
    elif dataset_type == 'original':
        dataset = DroneDatasetTiled(tile_size=tile_size, is_mosaic=True, transform=transform)
    elif dataset_type == 'upsample':
        dataset = DroneUpsampleTiled(f=f, r=r, t=t, height=height, tile_size=tile_size, is_mosaic=True, transform=transform, force_tiling=force_tiling)
        
    lengths = [int(len(dataset)*0.8), len(dataset)-int(len(dataset)*0.8)]
    trainset, testset = random_split(dataset, lengths)
    
    train_loader = DataLoader(trainset, batch_size=batch_size,shuffle=True)
    test_loader = DataLoader(testset, batch_size=batch_size)
    
    return train_loader, test_loader


def create_emulated_images(
                        f=1.644, 
                        r=0.5,
                        t=180e-6, 
                        height=600e3,
                        mask_threshold=0.2, 
                        in_path='./data/drone/original', 
                        out_path='./data/drone/emulated', 
                        device='cuda:0',
                        force_generation=False):
    
        
    if os.path.exists('data/drone/emulated') and force_generation:
        shutil.rmtree('data/drone/emulated')
    os.makedirs('data/drone/emulated/images')
    os.makedirs('data/drone/emulated/masks')
    
    print(f'Generating Images degradated with f = {f} [m], r = {r} [m], t = {t*1e6} [mus]\n')
    
    degradation = ImageDegradation(
        emulated_focal_length = f,
        emulated_radius = r,
        emulated_height = height,
        emulated_exposure_time = t)

    f=degradation.emulated_focal_length

    augmentation_scale = ComposeState([
        T.ToTensor(),
        T.Lambda(lambda x: x.to(device)),
        degradation
    ])

    dataset = DatasetSegmentation(Path(in_path, 'images'), Path(in_path, 'masks'), transform=augmentation_scale)

    imgs_names = dataset.images

    new_path_images, new_path_masks = list(map(lambda x: Path(out_path, x), ('images','masks')))
    new_path_images.mkdir(parents=True, exist_ok=True)
    new_path_masks.mkdir(parents=True, exist_ok=True)

    for i, (img,mask) in tqdm(enumerate(dataset)):
        img = np.array(img.cpu()[0]).astype(np.uint16)      
        fname =  Path(imgs_names[i]).name.replace('.DNG','')
        fname = fname+f'_f{f:.3f}_r{r:.2f}_t{int(t*1e6):04d}'.replace('.','_')+'.DNG'
        dng_from_template(
                img[::2,::2], img, imgs_names[i], Path(new_path_images,fname)
                )
        mask = (np.array(mask.cpu()[0]) > mask_threshold)
        fname =  fname.replace('DNG','png')
        mask = Image.fromarray(mask)
        mask.save(Path(new_path_masks,fname))       


def create_tiles_segmentation(dataset, img_dir, mask_dir, dataset_ref=None, tile_size=256):
    for folder in [img_dir, mask_dir]:
        if not os.path.exists(folder):
            os.makedirs(folder)
    for n, _ in enumerate(dataset):
        if dataset_ref is not None:
            (img_ref, mask_ref) = dataset_ref[n]
            tiled_img_ref = image2patches(img_ref, patch_size = (tile_size,tile_size), view_as_patches=True)
            tiled_mask_ref = image2patches(mask_ref, patch_size = (tile_size,tile_size), view_as_patches=True)
            *_, indices_without_class = class_detection(tiled_img_ref, tiled_mask_ref, None)  # Remove images without cars in it
        else:
            indices_without_class=None
        
        (img, mask) = dataset[n]
        tiled_img = image2patches(img, patch_size = (tile_size,tile_size), view_as_patches=True)
        tiled_mask = image2patches(mask, patch_size = (tile_size,tile_size), view_as_patches=True)
        tiled_img, tiled_mask, indices_without_class = class_detection(tiled_img, tiled_mask, indices_without_class)  
        
        for i, (sub_img, sub_mask) in enumerate(zip(tiled_img, tiled_mask)):
            tile_name = f"tile{n:04d}_{i:05d}"
            # dng_from_template(
            #     sub_img[::2,::2], 
            #     sub_img, 
            #     dataset.images[n], 
            #     Path(img_dir,tile_name))
            mask = (sub_mask > 0.2)
            # tile_name =  tile_name.replace('DNG','png')
            # mask = Image.fromarray(mask)
            # mask.save(Path(mask_dir,tile_name))   
            Image.fromarray(sub_img).save(os.path.join(img_dir, tile_name + '.tiff'))    
            Image.fromarray(sub_mask).save(os.path.join(mask_dir, tile_name + '.tiff'))


def class_detection(X, Y, without_class=None):
    """Split dataset in images which has the class in the target

       Args:
            X (ndarray): input image
            Y (ndarray): target with segmentation map (images with {0,1} values where it is 1 when there is the class)
       Returns:
           X_with_class (ndarray): input regions with the selected class 
           Y_with_class (ndarray): target regions with the selected class 
           X_without_class (ndarray): input regions without the selected class 
           Y_without_class (ndarray): target regions without the selected class 
    """

    if without_class is None:
        with_class = []
        without_class = []
        for i, img in enumerate(Y):
            if img.mean() == 0:
                without_class.append(i)
            else:
                with_class.append(i)

    X_with_class = np.delete(X, without_class, 0)
    Y_with_class = np.delete(Y, without_class, 0)

    return X_with_class, Y_with_class, without_class