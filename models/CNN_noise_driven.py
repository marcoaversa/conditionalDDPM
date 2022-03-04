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

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

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
    
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

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
    def __init__(self, dim, dim_out, *, groups = 8):
        super().__init__()

        self.block1 = Block(dim, dim_out)
        self.block2 = Block(dim_out, dim_out)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x):
        h = self.block1(x)
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

class CNNDriven(nn.Module):
    def __init__(
        self,
        dim,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        groups = 8,
        channels = 3,
    ):
        super().__init__()
        self.channels = channels
        
        self.blocks = nn.ModuleList([])

        dims = [channels, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1],dims[1:]))

        num_resolutions = len(in_out)
        
        for ind, (dim_in, dim_out) in enumerate(in_out):
            if ind >0:
                dim_in += channels
            self.blocks.append(nn.ModuleList([
                                ResnetBlock(dim_in, dim_out*2),
                                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                                ResnetBlock(dim_in, dim_out),
                                ResnetBlock(dim_out, dim_out),
                            ]))


            
        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(
            Block(dim+channels, dim),
            nn.Conv2d(dim, out_dim, 1)
        )

    def forward(self, x):
        
        for noise_sampler, attn, resnet, resnet2 in self.blocks:
            print(x.shape)
            gauss = noise_sampler(x)
            mu = gauss[:,:len(gauss[0])//2]
            sigma = gauss[:,len(gauss[0])//2:]
            x_att = attn(x)
            print(x.shape)
            x = resnet(x)
            print(x.shape)
            x = resnet2(x)
            print(x.shape)
            x = (x-mu) + torch.randn_like(x)*sigma
            print(x.shape)
            x = torch.cat((x, x_att), dim=1)
            print(x.shape)
        x = self.final_conv(x)

        return x
    
def train(self):
    backwards = partial(loss_backwards, self.fp16)

    avg_loss = [0,]

    while self.step < self.train_num_steps:

        for imgs, labels in self.train_loader:
            x=imgs.to(self.device)
            y = None if self.model_type == 'unc' else labels.to(self.device)
            loss = self.model(x,y)
            backwards(loss / self.gradient_accumulate_every, self.opt)
            if self.step != 0 and self.step % self.save_loss_every == 0:
                avg_loss[-1] /= self.save_loss_every
                print(f'Step: {self.step} - Loss: {avg_loss[-1]:.3f}')
                avg_loss.append(0)

            avg_loss[-1] += loss.item()

            self.opt.step()
            self.opt.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step != 0 and self.step % self.save_and_sample_every == 0:
                imgs, labels = next(iter(self.valid_loader))
                x_val=imgs.to(self.device)
                y_val = None if self.model_type == 'unc' else labels.to(self.device)
                milestone = self.step // self.save_and_sample_every
                n_images_to_sample = 25 
                batches = num_to_groups(n_images_to_sample, self.batch_size) 
                all_images_list = list(map(lambda n: self.ema_model.sample(y_val if y_val == None else y_val[:n], batch_size=n), batches))    
                all_images = torch.cat(all_images_list, dim=0)
                # all_images = (all_images + 1.)*0.5
                self.save_grid(all_images, str(self.results_folder / f'{milestone:03d}-train-pred.png'), nrow = 5)
                self.save(avg_loss)
                if self.model_type == 'c':
                    if len(y.shape) == 1:
                        self.save_with_1Dlabels(milestone, y_val, mode='train') 
                    else:
                        self.save_with_2Dlabels(milestone, x_val, batches, mode='train', var_type='original')
                        self.save_with_2Dlabels(milestone, y_val, batches, mode='train', var_type='condition')

            self.step += 1

    print('training completed')