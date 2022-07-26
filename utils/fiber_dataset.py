import h5py

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.transforms.functional import rotate
import torch.nn.functional as F

from random import random

from .augmentation import ComposeState


def import_fiber(dataset_name='MNIST', batch_size=32):
    
    assert dataset_name == 'MNIST' or dataset_name == 'Fashion-MNIST' or dataset_name == 'Random Patterns'
    
    loader=[]
    for mode in ['Training', 'Testing']:
        hf = h5py.File('/nfs/conditionalDDPM/data/fiber/Data_1m.h5' , 'r')
        
        speckles = hf[f'{mode}/Speckle_images/{dataset_name}']
        original = hf[f'{mode}/Original_images/{dataset_name}']

        speckles = np.array(speckles)
        original = np.array(original)
        hf.close()

        normalize = lambda x: (x-x.mean())/x.std()
#         normalize = lambda x: x/255.

        #Prepare speckles
        image_dim = 224
        speckles = np.reshape(speckles, (speckles.shape[0],image_dim,image_dim))[:,:,:,np.newaxis]
        speckles = torch.Tensor(speckles).permute(0,3,1,2).cuda()
#         speckles = F.avg_pool2d(speckles, 4)
#         speckles = normalize(speckles)
        speckles = speckles/85.

        #Prepare original
        original = torch.clip(F.interpolate(torch.Tensor(original).permute(0,3,1,2), speckles.shape[-1]).cuda(),0.,1.)
#         original = torch.Tensor(original).permute(0,3,1,2).cuda()
        original = normalize(original)

        rotations=[0,90,180,270] 
        rotation = lambda x: rotate(x[None], rotations[int(random()*len(rotations))])[0]

#         transform = ComposeState([
#             transforms.RandomVerticalFlip(),
#             transforms.RandomHorizontalFlip(),
#             rotation,
#                 ])       

        transform=None

        transform = transform if mode == 'Training' else None
        shuffle = True if mode == 'Training' else False

        dataset = FiberDataset([original, speckles], transform=transform, size=image_dim)

        loader.append(DataLoader(dataset, batch_size=batch_size, shuffle=shuffle))
        
        if dataset_name == 'Random Patterns':
            
            train_size = int(len(original)*0.8)
            
            train_set = FiberDataset([original[:train_size], speckles[:train_size]], transform=transform, size=image_dim)
            test_set = FiberDataset([original[train_size:], speckles[train_size:]], transform=transform, size=image_dim)
            
            return DataLoader(train_set, batch_size=batch_size, shuffle=True), DataLoader(test_set, batch_size=batch_size, shuffle=True)

    return loader[0], loader[1]

class FiberDataset(Dataset):
    """TensorDataset with support of transforms."""
    def __init__(self, tensors, transform=None, size: int = 224):
        assert all(tensors[0].shape[0] == tensor.shape[0] for tensor in tensors)
        self.tensors = tensors
        self.transform = transform
        self.size = size

    def __getitem__(self, index):
        x = self.tensors[0][index].clone()
        y = self.tensors[1][index].clone()
        
        if self.transform:
            x,y = self.transform([x,y])
            
        return x, y

    def __len__(self):
        return self.tensors[0].shape[0]