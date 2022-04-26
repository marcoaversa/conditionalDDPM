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

from utils.dataset import import_dataset

import argparse

import mlflow

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import MLFlowLogger
from mlflow.tracking.client import MlflowClient

os.umask(0o002)

parser = argparse.ArgumentParser(description='ConditionalDDPM')

# MLFlow

parser.add_argument('--tracking_uri', type=str, default='http://mlflowroute-2587054aproject.ida.dcs.gla.ac.uk/', help='MLFlow tracking URI')
parser.add_argument('--experiment_name', type= str, default='dev', help='experiment name tracked on mlflow server')
parser.add_argument('--run_name', type=str, default='test', help='run name tracked on mlflow')
parser.add_argument('--test_name', type=str, default='test', help='extra name for the test run')
parser.add_argument('--skip_sample', type=int, default=1, help='Sample every N steps in the DDPM sampling process')
parser.add_argument('--clip', type=float, default=1., help='Sample every N steps in the DDPM sampling process')

# Set Parser

args = parser.parse_args()

# Set MLFlow
client = MlflowClient(args.tracking_uri)
experiment_id = client.get_experiment_by_name(args.experiment_name).experiment_id
runs_names = [client.get_run(run.run_id).data.params['run_name'] for run in client.list_run_infos(experiment_id)]
runs_id = [run.run_id for run in client.list_run_infos(experiment_id)]
run_id = runs_id[runs_names.index(args.run_name)] # Return the id corresponding to the last run with that name
run = client.get_run(run_id)

integers = ['image_size','dim', 'n_layers', 'save_loss_every', 'sample_every', 'timesteps', 'train_num_steps', 'batch_size']
floats = ['lr']

for param in run.data.params:
    value = run.data.params[param]
    if param in integers:
        value = int(value)
    elif param in floats:
        value = float(value)
    globals()[param] = value
    
local_path = client.download_artifacts(run_id, "model_best.pt", f'/nfs/conditionalDDPM/tmp')

# Define Dataset

train_loader, valid_loader = import_dataset(data_name = dataset, 
                                            batch_size = batch_size)

x,y = next(iter(train_loader))

if dataset.startswith('ls'):
    if dataset.endswith('full'):
        _, channels, steps, height, width = x.shape
    else:
        _, channels, height, width = x.shape
else:
    _, channels, height, width = x.shape
assert height == width, 'Image should be square'
image_size = height
condition_dim = 1 if y.ndim == 1 else channels
dim_mults = [2**i for i in range(n_layers)]

# Define Model

"""Define NN"""
model = Unet(
    dim = dim,
    dim_mults = dim_mults,
    channels = channels if model_type == 'unc' else channels+condition_dim,
    out_dim = channels, 
)

"""Define DDPM"""
diffusion = GaussianDiffusion(
                model,
                image_size = image_size,
                timesteps = timesteps,  
                sample_every = args.skip_sample, 
                loss_type = loss,  
                channels = channels,
                model_type = model_type,
                clip = args.clip,
                device = device
                )

"""Define Pytorch Lightning Model"""
model = LitModelDDPM( 
                diffusion_model = diffusion, 
                model_type = model_type,
                batch_size = batch_size,
                lr = lr,
                save_loss_every = save_loss_every
                )

"""Save Results at the end of the training"""
model.model.denoise_fn.load_state_dict(torch.load(f'/nfs/conditionalDDPM/tmp/model_best.pt'))

"""Define Logger"""
mlf_logger = MLFlowLogger(tracking_uri=args.tracking_uri,
                          experiment_name=args.experiment_name, 
                          run_name=f'{args.run_name}_{args.test_name}'
                         )

run_id = mlf_logger.run_id

for key in list(args.__dict__.keys()):
    if key != 'run_name':
        mlf_logger.experiment.log_param(run_id=run_id, key=key, value=getattr(args, key))
mlf_logger.experiment.log_param(run_id=run_id, key='run_name', value=f'{args.run_name}_{args.test_name}')


"""Define Trainer"""
trainer = Trainer(
                 enable_checkpointing=False,
                 logger=mlf_logger,
                 max_steps = train_num_steps,
                 limit_val_batches=1,
                 limit_test_batches=1,
                 log_every_n_steps=50,
                 num_sanity_val_steps=0,
                 gpus=1 if device == 'cuda:0' else 0)

trainer.test(
        model,
        dataloaders=valid_loader
)