"""Copyright (c) 2020 Phil Wang"""

import os
import math
import copy
import torch
from torch import nn, einsum
import torch.nn.functional as F
from inspect import isfunction
from functools import partial

from torch.utils import data
from pathlib import Path
from torch.optim import Adam
from torchvision import transforms, utils
from PIL import Image

import matplotlib.pyplot as plt

import imageio

import numpy as np
from tqdm import tqdm
from einops import rearrange

try:
    from apex import amp
    APEX_AVAILABLE = True
except:
    APEX_AVAILABLE = False

# helpers functions

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def cycle(dl):
    while True:
        for data in dl:
            yield data

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def loss_backwards(fp16, loss, optimizer, **kwargs):
    if fp16:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward(**kwargs)
    else:
        loss.backward(**kwargs)

# small helper modules

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose2d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)

class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        std = torch.var(x, dim = 1, unbiased = False, keepdim = True).sqrt()
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (std + self.eps) * self.g + self.b

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

# building block modules

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim_out, 3, padding=1),
            nn.GroupNorm(groups, dim_out),
            Mish()
        )
    def forward(self, x):
        return self.block(x)

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, alpha_emb_dim = None, groups = 8):
        super().__init__()
        self.mlp_time = nn.Sequential(
            Mish(),
            nn.Linear(time_emb_dim, dim_out)
        ) if exists(time_emb_dim) else None
        
        self.mlp_alpha = nn.Sequential(
            Mish(),
            nn.Linear(alpha_emb_dim, dim_out)
        ) if exists(alpha_emb_dim) else None

        self.block1 = Block(dim, dim_out)
        self.block2 = Block(dim_out, dim_out)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb, alpha_emb):
        h = self.block1(x)

        if exists(self.mlp_time):
            h += self.mlp_time(time_emb)[:, :, None, None]
            
        if exists(self.mlp_alpha):
            h += self.mlp_alpha(alpha_emb)[:, :, None, None]

        h = self.block2(h)
        return h + self.res_conv(x)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)
        k = k.softmax(dim=-1)
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)

# model

class Unet(nn.Module):
    def __init__(
        self,
        dim,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        groups = 8,
        channels = 3,
        with_time_emb = True,
        with_alpha_emb = True
    ):
        super().__init__()
        self.channels = channels

        dims = [channels, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        if with_time_emb:
            time_dim = dim
            self.time_mlp = nn.Sequential(
                SinusoidalPosEmb(dim),
                nn.Linear(dim, dim * 4),
                Mish(),
                nn.Linear(dim * 4, dim)
            )
        else:
            time_dim = None
            self.time_mlp = None
            
        if with_alpha_emb:
            alpha_dim = dim
            self.alpha_mlp = nn.Sequential(
                SinusoidalPosEmb(dim),
                nn.Linear(dim, dim * 4),
                Mish(),
                nn.Linear(dim * 4, dim)
            )
        else:
            alpha_dim = None
            self.alpha_mlp = None

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                ResnetBlock(dim_in, dim_out, time_emb_dim = time_dim, alpha_emb_dim = alpha_dim),
                ResnetBlock(dim_out, dim_out, time_emb_dim = time_dim, alpha_emb_dim = alpha_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim = time_dim, alpha_emb_dim = alpha_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim)))
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim = time_dim, alpha_emb_dim = alpha_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                ResnetBlock(dim_out * 2, dim_in, time_emb_dim = time_dim, alpha_emb_dim = alpha_dim),
                ResnetBlock(dim_in, dim_in, time_emb_dim = time_dim, alpha_emb_dim = alpha_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Upsample(dim_in) if not is_last else nn.Identity()
            ]))

        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(
            Block(dim, dim),
            nn.Conv2d(dim, out_dim, 1)
        )

    def forward(self, x, time, alpha):
        t = self.time_mlp(time) if exists(self.time_mlp) else None
        alpha = self.alpha_mlp(alpha) if exists(self.alpha_mlp) else None

        h = []

        for resnet, resnet2, attn, downsample in self.downs:
            x = resnet(x, t, alpha)
            x = resnet2(x, t, alpha)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t, alpha)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t, alpha)

        for resnet, resnet2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, t, alpha)
            x = resnet2(x, t, alpha)
            x = attn(x)
            x = upsample(x)

        return self.final_conv(x)

# gaussian diffusion trainer class

def extract_pair(seq, t):
    x = torch.stack([s.squeeze()[t[i]+1] for i,s in enumerate(seq)])
    x = x[:,None] if x.ndim == 3 else x
    
    y = torch.stack([s.squeeze()[t[i]] for i,s in enumerate(seq)])
    y = y[:,None] if y.ndim == 3 else y
    return x, y

class TimeEvolver(nn.Module):
    def __init__(
        self,
        denoise_fn,
        *,
        timesteps = 19,
        n_alphas = 100,
        loss_type = 'l2',
        device='cuda'
    ):
        super().__init__()
        self.denoise_fn = denoise_fn
        
        self.num_timesteps = int(timesteps)
        self.num_alphas = n_alphas
        
        self.loss_type = loss_type
        
        self.device=device

        """Alpha between each step"""
    @torch.no_grad()
    def sample(self, x):
        b,c,h,w = x.shape
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            t = torch.full((b,), i, dtype=torch.long, device=self.device)
            alpha = torch.full((b,), 1., device=self.device)
            drift = self.denoise_fn(x, t, alpha)
#             x = (x-drift) + torch.sqrt(torch.clip(x-drift,0.,None)).mean()*torch.randn_like(x)
            x = x-drift
        return x

    """With Alpha"""
    def p_losses(self, x_seq, t, alphas):

        x,y = extract_pair(x_seq,t)
        sub_step = torch.einsum('i,ijkl->ijkl', alphas, x)+torch.einsum('i,ijkl->ijkl', (1.-alphas), y)
        drift = self.denoise_fn(sub_step, t, alphas)
        if self.loss_type == 'l1':
            loss = (drift - (y-sub_step)).abs().mean()
        elif self.loss_type == 'l2':
            loss = F.mse_loss(drift, (y-sub_step))
        else:
            raise NotImplementedError()
        return loss
    @torch.no_grad()

    def forward(self, x):
        b,c,seq,h,w = x.shape
        t = torch.randint(0, self.num_timesteps-1, (b,), device=x.device).long()
        alphas = torch.rand((b,), device=x.device)
        return self.p_losses(x, t, alphas)


# trainer class

class Trainer(object):
    def __init__(
        self,
        model,
        train_loader,
        valid_loader,
        *,
        train_batch_size = 32,
        train_lr = 2e-5,
        train_num_steps = 100000,
        save_and_sample_every = 1000,
        results_folder = './logs',
        device = 'cuda'
    ):
        super().__init__()
        
        self.model = model      

        self.save_and_sample_every = save_and_sample_every
        self.save_loss_every = 50

        self.batch_size = train_batch_size
        self.train_num_steps = train_num_steps

        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.opt = Adam(model.parameters(), lr=train_lr)
        
        self.step = 0

        self.device = device

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True, parents=True)

    def save(self, loss):
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'loss': loss,
        }
        torch.save(data, str(self.results_folder / f'model.pt'))
        with open(str(self.results_folder / f'loss.txt'), 'w') as file:
            for element in loss:
                file.write(str(element) + "\n")

    def load(self):
        data = torch.load(str(self.results_folder / f'model.pt'))

        self.step = data['step']
        self.model.load_state_dict(data['model'])
        
    def save_fig(self, x, y, pred, file_name):
        
        _, (ax0,ax1,ax2) = plt.subplots(1,3)
        ax0.imshow(x)
        ax0.set_title('Noisy')
        ax1.imshow(y)
        ax1.set_title('Clean')
        ax2.imshow(pred)
        ax2.set_title('Prediction')
        plt.savefig(file_name)

    def train(self):

        avg_loss = [0,]

        while self.step < self.train_num_steps:

            for _, x in self.train_loader:
                
                self.opt.zero_grad()
                
                loss = self.model(x.to(self.device))
                loss.backward()
                
                if self.step != 0 and self.step % self.save_loss_every == 0:
                    avg_loss[-1] /= self.save_loss_every
                    print(f'Step: {self.step} - Loss: {avg_loss[-1]:.3f}')
                    avg_loss.append(0)

                avg_loss[-1] += loss.item()

                self.opt.step()

                if self.step != 0 and self.step % self.save_and_sample_every == 0:
                    milestone = self.step // self.save_and_sample_every
                    self.save(avg_loss)
                    self.save_fig(x = x[0,0,-1].detach().cpu(),
                                  y = x[0,0,0].detach().cpu(), 
                                  pred = self.model.sample(x[:,:,-1].to(self.device))[0,0].detach().cpu(),
                                  file_name = str(self.results_folder / f'result_{milestone:03d}.png') )
                
                self.step += 1
        print('training completed')