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

    
def import_dataset(data_name: str = 'MNIST', batch_size: int = 32, image_size: int = 28,
                   sum_from: int = 0, sum_to: int = 50, import_timeseries = False, 
                   sum_every_n_steps: int = 5, seq_random: bool = True, force_download: bool = False):
    if data_name == 'MNIST':
        train_loader, valid_loader, image_size, channels, dim_mults = import_mnist(batch_size)
    elif data_name == 'CIFAR10':
        train_loader, valid_loader, image_size, channels, dim_mults = import_cifar10(batch_size)
    elif data_name == 'speckles':
        train_loader, valid_loader, image_size, channels, dim_mults = import_speckles(sum_from = sum_from, sum_to = sum_to, 
                                                                                      batch_size = batch_size, image_size = image_size,
                                                                                      import_timeseries = import_timeseries, 
                                                                                      sum_every_n_steps = sum_every_n_steps)
    elif data_name == 'light_sheets_full':
        train_loader, valid_loader, image_size, channels, dim_mults = import_ls(mode = 'full', 
                                                                                batch_size = batch_size, 
                                                                                image_size = image_size,
                                                                                force_download = force_download)
    elif data_name == 'light_sheets_ae':
        train_loader, valid_loader, image_size, channels, dim_mults = import_ls(mode = 'ae', 
                                                                                batch_size = batch_size, 
                                                                                image_size = image_size,
                                                                                force_download = force_download)
    elif data_name == 'light_sheets_seq':
        train_loader, valid_loader, image_size, channels, dim_mults = import_ls(mode = 'seq', 
                                                                                batch_size = batch_size, 
                                                                                image_size = image_size,
                                                                                seq_random = seq_random,
                                                                                force_download = force_download)

    return train_loader, valid_loader, image_size, channels, dim_mults

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
    return train_loader, valid_loader, image_size, channels, dim_mults

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
    return train_loader, valid_loader, image_size, channels, dim_mults


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
    return train_loader, valid_loader, image_size, channels, dim_mults


def import_speckles(sum_from: int = 0, sum_to: int = 10, batch_size: int = 32, 
                    image_size: int = 28, import_timeseries = False, sum_every_n_steps = 5):
    data_path = './data/speckles'
    mu = (0,)
    sigma = (1,)
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
    return train_loader, valid_loader, image_size, channels, dim_mults


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


def import_ls(mode: str = 'full', batch_size: int = 32, image_size: int = 128, seq_random: bool = True,force_download: bool = False):
    """Lightsheets dataset
       
       Args:
           -mode = different X,Y light sheet pairs
               -full = Y first stack image, X all diffusions for Y. On X we have an additional map with the level of depth
               -ae = X is equal to Y
    """
    data_path = './data/light_sheets'
    channels = 1
    dim_mults = (1,2,4)
    BG = 443
    noise_threshold = 20.
    
    if not os.path.exists(os.path.join(data_path, f'X_{mode}_{image_size}.pt')) or force_download:
    
        if os.path.exists(os.path.join(data_path, f'X_{mode}_{image_size}.pt')):
            os.remove(os.path.join(data_path, f'X_{mode}_{image_size}.pt'))
        if os.path.exists(os.path.join(data_path, f'Y_{mode}_{image_size}.pt')):
            os.remove(os.path.join(data_path, f'Y_{mode}_{image_size}.pt'))

        positions, z_stacks, x_shifts = detect_sequence(image_size = image_size, BG = BG)

        seq, delta_zs=[],[]
        for i in range(101-max(z_stacks)):
            names = [f'./Pos{p:02d}/img_channel000_position{p:03d}_time000000000_z{z_stacks[j]+i:03d}.tif' for j,p in enumerate(positions)]
            seq.append([torch.clip(Tensor(tiff.imread(os.path.join(data_path, name)).astype(np.int32))-BG,0,None) for name in names])
            delta_zs.append([z_stacks[j]+i - z_stacks[0] for j,p in enumerate(positions[1:])])

        n_shifts = len(positions)
        
        X, Y, Delta_Z= [], [], []
                        
        print('\nTiling images')
        
        if mode == 'full' or mode == 'ae':
            for i in tqdm(range(101-max(z_stacks))):
                for n in range(n_shifts-1):
                    x_image = torch.clip(seq[i][n+1], 0, None)
                    tiles_x = tile_image(x_image[:,:-x_shifts[n+1]], image_size)
                    X.append(tiles_x)

                    y_image = torch.clip(seq[i][0], 0, None)
                    tiles_y = tile_image(y_image[:,x_shifts[n+1]:], image_size)
                    Y.append(tiles_y)  

                    Delta_Z.append([delta_zs[i][n] for j in range(len(tiles_x))])
                    
            if mode == 'full':
                Z=[]
                for stack in Delta_Z:
                    Z += stack

                Z = [torch.ones((image_size,image_size))*z for z in Z]

                X = torch.cat(X)
                Y = torch.cat(Y)
                Z = torch.stack(Z)

                # Permute images
                b=torch.randperm(len(X))
                X = torch.cat([X[b, None], Z[b,None]], dim=1)
                Y = Y[b, None] 
        #         Z = Tensor(Z)[b].tolist()

            elif mode == 'ae':

                X = torch.cat(X)
                Y = torch.cat(Y)
                X = torch.cat([X,Y])

                # Permute images
                b=torch.randperm(len(X))
                X = X[b, None]
                Y = X
                
        elif mode == 'seq':
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
            Y = Y[indices, None]


        print('Saving Tiles')
        # Remove noisy images
        means = Tensor([img.mean().item() for img in Y])
        indices = means > noise_threshold
        Y = Y[indices]
        X = X[indices]

        torch.save(X, os.path.join(data_path, f'X_{mode}_{image_size}.pt'))
        torch.save(Y, os.path.join(data_path, f'Y_{mode}_{image_size}.pt'))
#         with open(os.path.join(data_path, f"deltaZ_{image_size}.txt"), "w") as f:
#             for element in Z:
#                 f.write(f'{int(element)}\n')
        print(f"\nDataset containes {len(X)} tiles")
        
    else:
        print("Loading Dataset")
        X = torch.load(os.path.join(data_path, f'X_{mode}_{image_size}.pt'))
        Y = torch.load(os.path.join(data_path, f'Y_{mode}_{image_size}.pt'))
        
    train_size = int(len(X)*0.8)
    
    mu,sigma = (X[:train_size].mean().item(),), (X[:train_size].std().item(),)

    train_set = LightSheetsDataset( (Y[:train_size], X[:train_size]), mode=mode, seq_random=seq_random)
    valid_set = LightSheetsDataset( (Y[train_size:], X[train_size:]), mode=mode, seq_random=seq_random)
    train_loader, valid_loader = set_dataloaders(train_set, valid_set, batch_size)
    print('Light Sheet data imported!')
    return train_loader, valid_loader, image_size, channels, dim_mults


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
        x = self.tensors[0][index]
        y = self.tensors[1][index]        
        
        if self.transform:
            x = self.transform(x)
            y = Tensor(y)
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
    def __init__(self, tensors, mode:str = 'seq', seq_random: bool = True, transform=None):
        assert all(tensors[0].shape[0] == tensor.shape[0] for tensor in tensors)
        self.tensors = tensors
        self.transform = transform
        self.mode = mode
        self.seq_random = seq_random 

    def __getitem__(self, index):
        x = self.tensors[0][index]
        y = self.tensors[1][index]        
       
        if self.mode == 'seq':
            
            energy = y[0,0].mean()# --> y_shape = (1,N_sequences,H,W)
            y /= energy
            gt_min = y[0,0].min()
            gt_max = y[0,0].max()
            y = torch.clip(y-gt_min,0,None)/(gt_max-gt_min)
            
            x = y[:,0]
            if self.seq_random:
                index=torch.randint(0,len(y[0])-1,(1,)).item()
                y = y[:,index]
            else:
                y = y[:,-1]
            
#         if self.transform:
#             x = (x-x.min())/(x.max()-x.min())
#             if y.shape[0] == 2:
#                 y[0] = (y[0]-y[0].min())/(y[0].max()-y[0].min())
#             else:
#                 y = (y-y.min())/(y.max()-y.min())
               
#             x = self.transform(x)
#             y = self.transform(y)

        x = x.type(torch.FloatTensor)        
        y = y.type(torch.FloatTensor)

        return x, y

    def __len__(self):
        return self.tensors[0].shape[0]

if __name__=='__main__':        
    train_loader, valid_loader = import_dataset(args)


# def import_light_sheets(batch_size: int = 32):
    
#     data_path = './data/light_sheets'
#     image_size=128
#     channels = 1
#     dim_mults = (1,2,4)
    
#     imgs=[]
#     for i in reversed(range(1,46)):
#         imgs.append(Tensor(np.load(os.path.join(data_path, f'pos{i}_ims.npz'))['arr_0']))
#     Y = imgs[0][:,:,:128].unfold(1, 128, 128).reshape(-1,128,128)[:,None].repeat(43,1,1,1) # shape: (392,1,128,128)
#     imgs.pop(0) # discard gt from the imgs list
#     imgs = [img[:,:,i*20:(128+i*20)].unfold(1, 128, 128).reshape(-1,128,128) for i, img in enumerate(imgs)]   
#     imgs.pop(5) # discard pos39 image (not well acquired)
    
#     X = imgs[0]
#     imgs.pop(0)
#     for img in imgs:
#         X = torch.cat((X,img), dim = 0) 
#     X = X[:, None] # X final shape: (16856, 128, 128)
        
#     train_size = int(len(X)*0.8)
    
#     mu,sigma = (X[:train_size].mean().item(),), (X[:train_size].std().item(),)

#     train_transform, test_transform = set_transforms(mu,sigma)

#     train_set = LightSheetsDataset( (Y[:train_size], X[:train_size]), transform=train_transform)
#     valid_set = LightSheetsDataset( (Y[train_size:], X[train_size:]), transform=test_transform)
#     train_loader, valid_loader = set_dataloaders(train_set, valid_set, batch_size)
#     return train_loader, valid_loader, image_size, channels, dim_mults

# def import_light_sheets2(batch_size: int = 32, image_size: int = 128):
    
#     data_path = './data/light_sheets'
#     channels = 1
#     dim_mults = (1,2,4)
#     shift = 20
    
#     test_image = np.load(os.path.join(data_path, f'pos1_ims.npz'))['arr_0']
#     c,h,w = test_image.shape
#     n_shifts = (w//shift - image_size//shift) # max number of position that we can have to shift the left-hand side to the right-hand side
# #     n_pos = n_shifts if n_shifts <= 38 else 38 # 39th shift has bad quality
# #     n_pos = 2 # 2 positions are enough --> 35k images 128x128
#     n_pos = 3 # 15k images 256x256
    
#     if not os.path.exists(os.path.join(data_path, f'X_shifts_{n_pos}_imgsize_{image_size}.pt')):
#         imgs=[]
#         for i in range(1,n_pos+2):
#             imgs.append(Tensor(np.load(os.path.join(data_path, f'pos{i}_ims.npz'))['arr_0']))

#         # Image clearer on the left and noiser on the right
#         # Data collected with a shift on the left
#         X = imgs[:-1]
#         Y = imgs[1:]

#         _,_,w = X[0].shape

#         Xs, Ys= [], []
#         for j in range(len(X)):
#             print(f'\nElaborate image position {j+1}/{n_pos}')
#             for n in range(n_shifts):
#                 sys.stdout.write(f'\r Slice N: {n+1}/{n_shifts}')
#                 sys.stdout.flush()
#                 Xs.append(X[j][:,:,n*shift:(image_size+n*shift)].unfold(1, image_size, image_size).reshape(-1,image_size,image_size))
#                 Ys.append(Y[j][:,:,n*shift:(image_size+n*shift)].unfold(1, image_size, image_size).reshape(-1,image_size,image_size))

#         X = Xs[0]
#         print('\nStacking X tiles')
#         for img in tqdm(Xs):
#             X = torch.cat((X,img), dim = 0) 
#         X = X[:, None] 
#         torch.save(X, os.path.join(data_path, f'X_shifts_{n_pos}_imgsize_{image_size}.pt'))

#         Y = Ys[0]
#         print('\nStacking Y tiles')
#         for img in tqdm(Ys):
#             Y = torch.cat((Y,img), dim = 0) 
#         Y = Y[:, None] 
#         torch.save(Y, os.path.join(data_path, f'Y_shifts_{n_pos}_imgsize_{image_size}.pt'))
        
#         print(f"\nDataset containes {len(X)} tiles")
        
#     else:
#         print("Loading Dataset")
#         X = torch.load(os.path.join(data_path, f'X_shifts_{n_pos}_imgsize_{image_size}.pt'))
#         Y = torch.load(os.path.join(data_path, f'Y_shifts_{n_pos}_imgsize_{image_size}.pt'))
        
#     train_size = int(len(X)*0.8)
#     print(train_size)
    
#     mu,sigma = (X[:train_size].mean().item(),), (X[:train_size].std().item(),)

#     train_transform, test_transform = set_transforms(mu,sigma)

#     train_set = LightSheetsDataset( (Y[:train_size], X[:train_size]), transform=train_transform)
#     valid_set = LightSheetsDataset( (Y[train_size:], X[train_size:]), transform=test_transform)
#     train_loader, valid_loader = set_dataloaders(train_set, valid_set, batch_size)
#     print('Light Sheet data imported!')
#     return train_loader, valid_loader, image_size, channels, dim_mults

# def import_light_sheets3(batch_size: int = 32, image_size: int = 128):
    
#     data_path = './data/light_sheets'
#     channels = 1
#     dim_mults = (1,2,4)
#     shift = 40
    
#     test_image = tiff.imread(os.path.join(data_path, f'Pos00/img_channel000_position000_time000000000_z000.tif'))
#     h,w = test_image.shape
#     n_shifts = (w//shift - image_size//shift) # max number of position that we can have to shift the left-hand side to the right-hand side
# #     n_pos = n_shifts if n_shifts <= 38 else 38 # 39th shift has bad quality
# #     n_pos = 2 # 2 positions are enough --> 35k images 128x128
#     n_pos = 1 # 15k images 256x256
    
#     # After 10 micron shift, the focus of the light sheet goes deep in the sample. 
#     # It means that the ground truth is at d = 10*tg(pi/2) = 10 micron
#     # In the z-stack, the sample moves deeper 2.0 micron every step --> around 5 images later we have the gt
#     z_shift = 5
    
#     if not os.path.exists(os.path.join(data_path, f'X_shifts_{n_pos}_imgsize_{image_size}.pt')):
    
#         Xs, Ys= [], []
#         print('\nTiling images')
#         for n in range(n_pos):
#             dir_X = os.path.join(data_path, f'Pos{n+1:02d}')
#             dir_Y = os.path.join(data_path, f'Pos{n:02d}')
#             X,Y=[],[]
#             for i in range(101):
#                 x_image = tiff.imread(os.path.join(dir_X, f'img_channel000_position{n+1:03d}_time000000000_z{i:03d}.tif'))
#                 y_image = tiff.imread(os.path.join(dir_Y, f'img_channel000_position{n:03d}_time000000000_z{i:03d}.tif'))
#                 X.append(Tensor(x_image.astype(np.int32)))
#                 Y.append(Tensor(y_image.astype(np.int32)))

#             X = X[:-z_shift]
#             Y = Y[z_shift:]

#             h,w = x_image.shape

#             for j in range(len(X)):
# #                 print(f'\nElaborate Z images N {j+1}/{101}')
#                 for n in range(n_shifts):
# #                     sys.stdout.write(f'\r Slice N: {n+1}/{n_shifts}')
# #                     sys.stdout.flush()
#                     Xs.append(tile_image(X[j][:,n*shift:(image_size+n*shift)], image_size))
#                     Ys.append(tile_image(Y[j][:,n*shift:(image_size+n*shift)], image_size))

# #         Xs = [x if x.max for x in Xs]
        
#         X = torch.cat(Xs)
#         Y = torch.cat(Ys)
        
#         b=torch.randperm(len(X))
#         X = X[b, None] 
#         Y = Y[b, None] 
        
#         torch.save(X, os.path.join(data_path, f'X_shifts_{n_pos}_imgsize_{image_size}.pt'))
#         torch.save(Y, os.path.join(data_path, f'Y_shifts_{n_pos}_imgsize_{image_size}.pt'))
#         print(f"\nDataset containes {len(X)} tiles")
        
#     else:
#         print("Loading Dataset")
#         X = torch.load(os.path.join(data_path, f'X_shifts_{n_pos}_imgsize_{image_size}.pt'))
#         Y = torch.load(os.path.join(data_path, f'Y_shifts_{n_pos}_imgsize_{image_size}.pt'))
        
#     train_size = int(len(X)*0.8)
#     print(train_size)
    
#     mu,sigma = (X[:train_size].mean().item(),), (X[:train_size].std().item(),)

#     train_transform, test_transform = set_transforms(mu,sigma)

#     train_set = LightSheetsDataset( (Y[:train_size], X[:train_size]), transform=train_transform)
#     valid_set = LightSheetsDataset( (Y[train_size:], X[train_size:]), transform=test_transform)
#     train_loader, valid_loader = set_dataloaders(train_set, valid_set, batch_size)
#     print('Light Sheet data imported!')
#     return train_loader, valid_loader, image_size, channels, dim_mults

# def import_light_sheets_sequential(batch_size: int = 32, image_size: int = 128):
    
#     data_path = './data/light_sheets'
#     channels = 1
#     dim_mults = (1,2,4)
    
#     positions =   [ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14]
#     z_stacks =    [ 4, 11, 14, 18, 20, 24, 28, 30, 34, 38, 40, 45, 49, 54]
#     x_shifts =    [ 0, 54, 38, 35, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30]
    
#     images=[]
#     for i in range(4,101-max(z_stacks)):
#             names = [f'./Pos{p:02d}/img_channel000_position{p:03d}_time000000000_z{z_stacks[j]+i:03d}.tif' for j,p in enumerate(positions)]
#             images.append([tiff.imread(os.path.join(data_path, name)) for name in names])
    
#     n_shifts = len(positions)
    
#     if not os.path.exists(os.path.join(data_path, f'X_shifts_seq_imgsize_{image_size}.pt')):
    
#         X, Y= [], []
        
#         print('\nTiling images')
#         for i in range(101-max(z_stacks)-4):
#             for n in range(n_shifts-1):
#                 x_image = Tensor(images[i][n+1].astype(np.int32))
#                 tiles_x = tile_image(x_image[:,:-x_shifts[n+1]], image_size)
#                 X.append(tiles_x)

#                 y_image = Tensor(images[i][n].astype(np.int32))
#                 tiles_y = tile_image(y_image[:,x_shifts[n+1]:], image_size)
#                 Y.append(tiles_y)

#         X = torch.cat(X)
#         Y = torch.cat(Y)
        
#         b=torch.randperm(len(X))
#         X = X[b, None] 
#         Y = Y[b, None] 
        
#         torch.save(X, os.path.join(data_path, f'X_shifts_seq_imgsize_{image_size}.pt'))
#         torch.save(Y, os.path.join(data_path, f'Y_shifts_seq_imgsize_{image_size}.pt'))
#         print(f"\nDataset containes {len(X)} tiles")
        
#     else:
#         print("Loading Dataset")
#         X = torch.load(os.path.join(data_path, f'X_shifts_seq_imgsize_{image_size}.pt'))
#         Y = torch.load(os.path.join(data_path, f'Y_shifts_seq_imgsize_{image_size}.pt'))
        
#     train_size = int(len(X)*0.8)
#     print(train_size)
    
#     mu,sigma = (X[:train_size].mean().item(),), (X[:train_size].std().item(),)

#     train_transform, test_transform = set_transforms(mu,sigma)

#     train_set = LightSheetsDataset( (Y[:train_size], X[:train_size]), transform=train_transform)
#     valid_set = LightSheetsDataset( (Y[train_size:], X[train_size:]), transform=test_transform)
#     train_loader, valid_loader = set_dataloaders(train_set, valid_set, batch_size)
#     print('Light Sheet data imported!')
#     return train_loader, valid_loader, image_size, channels, dim_mults

# def import_light_sheets_ae(batch_size: int = 32, image_size: int = 128):
    
#     data_path = './data/light_sheets'
#     channels = 1
#     dim_mults = (1,2,4)
    
#     positions =   [ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14]
#     z_stacks =    [ 4, 11, 14, 18, 20, 24, 28, 30, 34, 38, 40, 45, 49, 54]
#     x_shifts =    [ 0, 54, 38, 35, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30]
    
#     images=[]
#     for i in range(4,101-max(z_stacks)):
#             names = [f'./Pos{p:02d}/img_channel000_position{p:03d}_time000000000_z{z_stacks[j]+i:03d}.tif' for j,p in enumerate(positions)]
#             images.append([tiff.imread(os.path.join(data_path, name)) for name in names])
    
#     n_shifts = len(positions)
    
#     if not os.path.exists(os.path.join(data_path, f'X_shifts_ae_imgsize_{image_size}.pt')):
    
#         X, Y= [], []
        
#         print('\nTiling images')
#         for i in range(101-max(z_stacks)-4):
#             for n in range(n_shifts-1):
#                 x_image = Tensor(images[i][n+1].astype(np.int32))
#                 tiles_x = tile_image(x_image[:,:-x_shifts[n+1]], image_size)
#                 X.append(tiles_x)

#                 y_image = Tensor(images[i][n].astype(np.int32))
#                 tiles_y = tile_image(y_image[:,x_shifts[n+1]:], image_size)
#                 Y.append(tiles_y)

#         X = torch.cat(X)
#         Y = torch.cat(Y)
        
#         b=torch.randperm(len(X))
#         X = X[b, None] 
#         Y = Y[b, None] 
        
#         torch.save(X, os.path.join(data_path, f'X_shifts_ae_imgsize_{image_size}.pt'))
#         torch.save(Y, os.path.join(data_path, f'Y_shifts_ae_imgsize_{image_size}.pt'))
#         print(f"\nDataset containes {len(X)} tiles")
        
#     else:
#         print("Loading Dataset")
#         X = torch.load(os.path.join(data_path, f'X_shifts_ae_imgsize_{image_size}.pt'))
#         Y = torch.load(os.path.join(data_path, f'Y_shifts_ae_imgsize_{image_size}.pt'))
        
#     train_size = int(len(X)*0.8)
#     print(train_size)
    
#     mu,sigma = (X[:train_size].mean().item(),), (X[:train_size].std().item(),)

#     train_transform, test_transform = set_transforms(mu,sigma)

#     train_set = LightSheetsDataset( (X[:train_size], X[:train_size]), transform=train_transform)
#     valid_set = LightSheetsDataset( (X[train_size:], X[train_size:]), transform=test_transform)
#     train_loader, valid_loader = set_dataloaders(train_set, valid_set, batch_size)
#     print('Light Sheet data imported!')
#     return train_loader, valid_loader, image_size, channels, dim_mults

