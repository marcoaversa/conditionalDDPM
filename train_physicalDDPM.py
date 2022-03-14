import os

from numpy import save
import matplotlib.pyplot as plt

from tqdm import tqdm

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from models.physicalDDPM import Unet, TimeEvolver, Trainer

from utils.dataset import import_dataset

import argparse

os.umask(0o002)

parser = argparse.ArgumentParser(description='ConditionalDDPM')

# Data & Results 
parser.add_argument('--dataset', type=str, default='MNIST', choices=('MNIST',
                                                                     'CIFAR10',
                                                                     'speckles',
                                                                     'light_sheets_full',
                                                                     'light_sheets_ae',
                                                                     'light_sheets_seq'), help='Choose the dataset')
parser.add_argument('--dataset_path', type=str, default='./data', help='Choose the dataset path')
parser.add_argument('--logdir', type=str, default='./logs', help='results path')
parser.add_argument('--save_sample_every', type=int, default=1000, help='Save and Sample after 1000 steps')

# Dataset Info

parser.add_argument('--image_size', type=int, default=28, help='Select input image size')
parser.add_argument('--sum_from', type=int, default=0, help='just for speckle dataset, start integration point')
parser.add_argument('--sum_to', type=int, default=50, help='just for speckle dataset, end integration point')
parser.add_argument('--import_timeseries', action='store_true', default=False,
                    help='just for speckle dataset, import the timeseries integrated every n steps')
parser.add_argument('--sum_every_n_steps', type=int, default=5, help='just for speckle dataset, time integrated every n steps (e.g. [0,5],[0,10],[0,15],...')

# Train Mode args
parser.add_argument('--n_alphas', type=int, default=100, help='Select number of alphas for each timestep')
parser.add_argument('--train_num_steps', type=int, default=100000, help='Number of training steps')
parser.add_argument('--loss', type=str, default='l2', choices=('l1','l2'), help='Select loss type')
parser.add_argument('--lr', type=float, default=2e-05, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=32, help='Select batch size')
parser.add_argument('--device', type=str, default='cuda', choices=('cpu','cuda'), help='Select the device')

# Visualization
parser.add_argument('--gif_type', type=str, default='sampling', choices=('sampling','histo'), help='Choose what kind of GIF to generate')


# Define arguments

args = parser.parse_args()

data_name = args.dataset
data_path = args.dataset_path
logdir = args.logdir
save_sample_every = args.save_sample_every

image_size = args.image_size
sum_from = args.sum_from
sum_to = args.sum_to
import_timeseries = args.import_timeseries
sum_every_n_steps = args.sum_every_n_steps

n_alphas = args.n_alphas
train_num_steps = args.train_num_steps
loss = args.loss
lr = args.lr
batch_size = args.batch_size
device = args.device

gif_type = args.gif_type

# Define Dataset

train_loader, valid_loader, image_size, channels, dim_mults = import_dataset(data_name, batch_size, image_size, sum_from, 
                                                                             sum_to, import_timeseries, sum_every_n_steps)

x,y = next(iter(valid_loader))

timesteps = y.shape[2]

# Define Model

model = Unet(
    dim = 64,
    dim_mults = dim_mults,
    channels = channels,
    out_dim = channels, 
)

model = model.to(device)

time_evolver = TimeEvolver(
                model,
                timesteps = timesteps,   # number of steps
                n_alphas = n_alphas,
                loss_type = loss,    # L1 or L2
                device = device
                )

trainer = Trainer( 
                model = time_evolver, 
                train_loader = train_loader, 
                valid_loader = valid_loader, 
                results_folder = logdir,
                train_batch_size = batch_size,
                save_and_sample_every = save_sample_every,
                train_lr = lr,
                train_num_steps = train_num_steps,   
                device = device
                )

trainer.train()