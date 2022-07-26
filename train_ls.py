import os

from numpy import save
import matplotlib.pyplot as plt

from tqdm import tqdm

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from models.DDPM_LearnSigma import Unet, GaussianDiffusion, LitModelDDPM
# from models.DDPMseq import Unet, GaussianDiffusion, LitModelDDPM
# from models.DDPMdp import Unet, GaussianDiffusion, LitModelDDPM

from utils.ls_dataset import import_ls

import argparse

import mlflow

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import MLFlowLogger

from utils.augmentation import ComposeState

os.umask(0o002)

parser = argparse.ArgumentParser(description='ConditionalDDPM')

# MLFlow

parser.add_argument('--tracking_uri', type=str, default='http://mlflowroute-2587054aproject.ida.dcs.gla.ac.uk/', help='MLFlow tracking URI')
parser.add_argument('--experiment_name', type= str, default='dev', help='experiment name tracked on mlflow server')
parser.add_argument('--run_name', type=str, default='test', help='run name tracked on mlflow')

# Data & Results 
# parser.add_argument('--dataset', type=str, default='MNIST', choices=('MNIST',
#                                                                      'CIFAR10',
#                                                                      'speckles',
#                                                                      'ls_random',
#                                                                      'ls_full',
#                                                                      'ls_firstlast',
#                                                                      'ls_ae',
#                                                                      'ls_aedp',), help='Choose the dataset')
parser.add_argument('--dataset', type=str, default='ls_full', help='Choose the dataset')
parser.add_argument('--dataset_path', type=str, default='./data', help='Choose the dataset path')
parser.add_argument('--image_size', type=int, default=128, help='Decide the size of the cropped images')

# Model

parser.add_argument('--dim', type=int, default=32, help='Choose the number of channels on the first UNet s layer')
parser.add_argument('--n_layers', type=int, default=4, help='Choose how many layers for the UNet')

# Mode
parser.add_argument('--mode', type=str, default='train', choices=('train','test'), help='Select mode')
parser.add_argument('--model_type', type=str, default='c', choices=('unc','c'), help='Select model type')
parser.add_argument('--save_loss_every', type=int, default=50, help='Save loss function every N steps')
parser.add_argument('--sample_every', type=int, default=1, help='Sample every N steps in the DDPM sampling process')

# Train Mode args
parser.add_argument('--timesteps', type=int, default=1000, help='Select number of timesteps diffusion model')
parser.add_argument('--train_num_steps', type=int, default=100000, help='Number of training steps')
parser.add_argument('--loss', type=str, default='l2', choices=('l1','l2','vae'), help='Select loss type')
parser.add_argument('--lr', type=float, default=2e-05, help='Learning rate')
parser.add_argument('--clip', type=float, default=1., help='Clipping during Sampling')
parser.add_argument('--batch_size', type=int, default=32, help='Select batch size')
parser.add_argument('--device', type=str, default='cuda:0', choices=('cpu','cuda:0'), help='Select the device')

# Set Parser

args = parser.parse_args()

# Set MLFlow

mlflow.set_tracking_uri(args.tracking_uri)
mlflow.set_experiment(args.experiment_name)

# Define Dataset

BG=99.6
mu=388.85
sigma=53.15

transform = ComposeState([
    transforms.RandomCrop(128),
#     transforms.Lambda(lambda x: (x/10852.0)),
    transforms.Lambda(lambda x: (x-BG-mu)/sigma),
#     transforms.GaussianBlur(3, sigma=(0.1, 2.0))
    ])

train_loader, valid_loader = import_ls(
                                        name=args.dataset, 
                                        batch_size=args.batch_size, 
                                        image_size = args.image_size,
                                        transform=transform,
                                        force_download = False)
x,y = next(iter(train_loader))

if args.dataset.startswith('ls'):
    if args.dataset.endswith('full'):
        _, channels, steps, height, width = x.shape
    else:
        _, channels, height, width = x.shape
else:
    _, channels, height, width = x.shape
assert height == width, 'Image should be square'
image_size = height
condition_dim = 1 if y.ndim == 1 else channels
dim_mults = [2**i for i in range(args.n_layers)]

# Define Model

"""Define NN"""
model = Unet(
    dim = args.dim,
    dim_mults = dim_mults,
    channels = channels if args.model_type == 'unc' else channels+condition_dim,
    out_dim = channels, 
)

"""Define DDPM"""
diffusion = GaussianDiffusion(
                model,
                image_size = image_size,
                timesteps = args.timesteps,  
                sample_every = args.sample_every, 
                loss_type = args.loss,  
                channels = channels,
                model_type = args.model_type,
                clip = args.clip,
                device = args.device
                )

"""Define Pytorch Lightning Model"""
model = LitModelDDPM( 
                diffusion_model = diffusion, 
                model_type = args.model_type,
                batch_size = args.batch_size,
                lr = args.lr,
                save_loss_every = args.save_loss_every
                )

"""Define Logger"""
mlf_logger = MLFlowLogger(tracking_uri=args.tracking_uri,
                          experiment_name=args.experiment_name, 
                          run_name=args.run_name
                         )

run_id = mlf_logger.run_id

for key in list(args.__dict__.keys()):
    mlf_logger.experiment.log_param(run_id=run_id, key=key, value=getattr(args, key))

"""Define Trainer"""
trainer = Trainer(
                 enable_checkpointing=False,
                 logger=mlf_logger,
                 max_steps = args.train_num_steps,
                 limit_val_batches=1,
                 limit_test_batches=1,
                 log_every_n_steps=50,
                 num_sanity_val_steps=0,
                 gpus=1 if args.device == 'cuda:0' else 0)

"""Train the model"""
trainer.fit(
        model,
        train_dataloaders=train_loader
)

"""Save Results at the end of the training"""
model.model.denoise_fn.load_state_dict(torch.load(f'/nfs/conditionalDDPM/tmp/model_best.pt'))

trainer.test(
        model,
        dataloaders=valid_loader
)