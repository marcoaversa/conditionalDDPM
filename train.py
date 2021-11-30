import os

from numpy import save
import matplotlib.pyplot as plt

from tqdm import tqdm

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from models.DDPM import Unet, GaussianDiffusion, Trainer

from utils.dataset import import_dataset

import argparse

os.umask(0o002)

parser = argparse.ArgumentParser(description='ConditionalDDPM')

# Data & Results 
parser.add_argument('--dataset', type=str, default='MNIST', choices=('MNIST','CIFAR10','speckles'), help='Choose the dataset')
parser.add_argument('--dataset_path', type=str, default='./data', help='Choose the dataset path')
parser.add_argument('--logdir', type=str, default='./logs', help='results path')
parser.add_argument('--save_sample_every', type=int, default=1000, help='Save and Sample after 1000 steps')

# Mode
parser.add_argument('--mode', type=str, default='train', choices=('train','test'), help='Select mode')
parser.add_argument('--model_type', type=str, default='conditional', 
                        choices=('unconditional','conditional'), help='Select model type')


# Train Mode args
parser.add_argument('--image_size', type=int, default=28, help='Select input image size')
parser.add_argument('--timesteps', type=int, default=1000, help='Select number of timesteps diffusion model')
parser.add_argument('--train_num_steps', type=int, default=100000, help='Number of training steps')
parser.add_argument('--loss', type=str, default='l2', choices=('l1','l2'), help='Select loss type')
parser.add_argument('--lr', type=float, default=2e-05, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=32, help='Select batch size')
parser.add_argument('--device', type=str, default='cuda', choices=('cpu','cuda'), help='Select the device')


# Define arguments

args = parser.parse_args()

data_name = args.dataset
data_path = args.dataset_path
logdir = args.logdir
save_sample_every = args.save_sample_every

mode = args.mode
model_type = args.model_type

image_size = args.image_size
timesteps = args.timesteps
train_num_steps = args.train_num_steps
loss = args.loss
lr = args.lr
batch_size = args.batch_size
device = args.device

# Define Dataset

train_loader, valid_loader, image_size, channels, dim_mults = import_dataset(data_name, batch_size)

# Define Model

model = Unet(
    dim = 64,
    dim_mults = dim_mults,
    channels = channels if model_type == 'unconditional' else channels+1,
    out_dim = channels, 
)

model = model.to(device)

diffusion = GaussianDiffusion(
model,
image_size = image_size,
timesteps = timesteps,   # number of steps
loss_type = loss,    # L1 or L2
channels = channels,
model_type = model_type,
device = device
)

# Training

trainer = Trainer( 
                    diffusion_model = diffusion, 
                    train_loader = train_loader, 
                    valid_loader = valid_loader, 
                    model_type = model_type,
                    results_folder = logdir,
                    save_and_sample_every = save_sample_every,
                    train_lr = lr,
                    train_num_steps = train_num_steps,   
                    device = device
                    )

if mode == 'train':
    trainer.train()

if mode == 'test':
    trainer.load()
    trainer.test()