import os

from numpy import save
import matplotlib.pyplot as plt

from tqdm import tqdm

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from models.DDPM_MLFlow import Unet, GaussianDiffusion, LitModelDDPM
from models.GBM import Unet, GBM, TrainerGBM

from utils.dataset import import_dataset
from utils.mlflow import display_mlflow_run_info

import argparse

import mlflow

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import MLFlowLogger

os.umask(0o002)

parser = argparse.ArgumentParser(description='ConditionalDDPM')

# MLFlow

parser.add_argument('--tracking_uri', type=str, default='http://mlflowroute-2587054aproject.ida.dcs.gla.ac.uk/', help='MLFlow tracking URI')
parser.add_argument('--experiment_name', type= str, default='dev', help='experiment name tracked on mlflow server')
parser.add_argument('--run_name', type=str, default='test', help='run name tracked on mlflow')

# Data & Results 
parser.add_argument('--dataset', type=str, default='MNIST', choices=('MNIST',
                                                                     'CIFAR10',
                                                                     'speckles',
                                                                     'light_sheets_full',
                                                                     'light_sheets_ae',
                                                                     'light_sheets_seq'), help='Choose the dataset')
parser.add_argument('--dataset_path', type=str, default='./data', help='Choose the dataset path')
parser.add_argument('--image_size', type=int, default=28, help='Select input image size')
parser.add_argument('--save_sample_every', type=int, default=1000, help='Save and Sample after 1000 steps')

# Mode
parser.add_argument('--mode', type=str, default='train', choices=('train','test'), help='Select mode')
parser.add_argument('--model_type', type=str, default='c', 
                        choices=('unc','c'), help='Select model type')

# Dataset Info


# Train Mode args
parser.add_argument('--timesteps', type=int, default=1000, help='Select number of timesteps diffusion model')
parser.add_argument('--train_num_steps', type=int, default=100000, help='Number of training steps')
parser.add_argument('--loss', type=str, default='l2', choices=('l1','l2'), help='Select loss type')
parser.add_argument('--lr', type=float, default=2e-05, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=32, help='Select batch size')
parser.add_argument('--device', type=str, default='cuda', choices=('cpu','cuda'), help='Select the device')

# Set Parser

args = parser.parse_args()

# Set MLFlow

mlflow.set_tracking_uri(args.tracking_uri)
mlflow.set_experiment(args.experiment_name)

# Define Dataset

train_loader, valid_loader, image_size, channels, dim_mults = import_dataset(args.dataset, args.batch_size, args.image_size)

condition_dim = 2 if args.dataset == 'light_sheets_full' else 1

# Define Model

model = Unet(
    dim = 64,
    dim_mults = dim_mults,
    channels = channels if args.model_type == 'unc' else channels+condition_dim,
    out_dim = channels, 
)

diffusion = GaussianDiffusion(
                model,
                image_size = args.image_size,
                timesteps = args.timesteps,   
                loss_type = args.loss,  
                channels = channels,
                model_type = args.model_type
                )

model = LitModelDDPM( 
                diffusion_model = diffusion, 
                model_type = args.model_type,
                batch_size = args.batch_size,
                lr = args.lr
                )

if args.mode == 'train':
#     with mlflow.start_run(run_name=args.run_name) as run:
#     for key in list(args.__dict__.keys()):
#         mlflow.log_param(key, getattr(args, key))

    mlf_logger = MLFlowLogger(tracking_uri=args.tracking_uri,
                              experiment_name=args.experiment_name, 
                              run_name=args.run_name
                             )
    
    run_id = mlf_logger.run_id

    for key in list(args.__dict__.keys()):
#             mlflow.log_param(key, getattr(args, key))
        mlf_logger.experiment.log_param(run_id=run_id, key=key, value=getattr(args, key))

    trainer = Trainer(
                     enable_checkpointing=False,
                     logger=mlf_logger,
                     max_steps = args.train_num_steps,
                     val_check_interval = args.save_sample_every,
                     log_every_n_steps=50,
                     gpus=1 if args.device == 'cuda' else 0)

    t = trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=valid_loader,
    )

#         display_mlflow_run_info(run)

#         trainer.train(run)

if args.mode == 'test':
    trainer.load()
    trainer.test()