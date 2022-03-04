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
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            Mish(),
            nn.Linear(time_emb_dim, dim_out)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out)
        self.block2 = Block(dim_out, dim_out)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        h = self.block1(x)

        if exists(self.mlp):
            h += self.mlp(time_emb)[:, :, None, None]

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
        with_time_emb = True
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

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                ResnetBlock(dim_in, dim_out, time_emb_dim = time_dim),
                ResnetBlock(dim_out, dim_out, time_emb_dim = time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim)))
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim = time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                ResnetBlock(dim_out * 2, dim_in, time_emb_dim = time_dim),
                ResnetBlock(dim_in, dim_in, time_emb_dim = time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Upsample(dim_in) if not is_last else nn.Identity()
            ]))

        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(
            Block(dim, dim),
            nn.Conv2d(dim, out_dim, 1)
        )

    def forward(self, x, time):
        t = self.time_mlp(time) if exists(self.time_mlp) else None

        h = []

        for resnet, resnet2, attn, downsample in self.downs:
            x = resnet(x, t)
            x = resnet2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for resnet, resnet2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, t)
            x = resnet2(x, t)
            x = attn(x)
            x = upsample(x)

        return self.final_conv(x)
    
class GBM(nn.Module):
    def __init__(
        self,
        model,
        image_size,
        channels = 3,
        subtimeseries_length = 3,
        timesteps = 500,
        loss_type = 'l1',
        device='cuda'
    ):
        super().__init__()
        self.model = model
        self.image_size = image_size
        self.channels = channels
        self.k = subtimeseries_length
        self.num_timesteps = timesteps
        self.loss_type = loss_type

        self.device=device

        
    def extract(self, a, t, x_shape):
        b, *_ = t.shape
        out = a.gather(-1, t)
        return out.reshape(b, *((1,) * (len(x_shape) - 1)))
        
    @torch.no_grad()
    def p_sample(self, x, t):
#         mu_sigma = self.propagator(x, t)
#         mu=mu_sigma[:,:self.channels]
#         sigma=mu_sigma[:,self.channels:]
#         diff = self.propagator(x,t)
#         x_t = self.time_evolution(x, t, mu, sigma)
#         return x_t
#         diff = self.model(x,t)
#         return self.time_evolution(x, diff)
        err = self.model(x, t)
        mu=err[:,:self.channels]
        sigma=err[:,self.channels:]
        return x-(mu+sigma*torch.randn_like(x))

    @torch.no_grad()
    def p_sample_loop(self, img, shape):
        b,_,h,w = shape
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            img = self.p_sample(img, torch.full((b,), i, dtype=torch.long, device=self.device))   
        return img

    @torch.no_grad()
    def sample(self, x, batch_size = 16):
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop(x, (batch_size, channels, image_size, image_size))
    
    def sub_timeseries(self, x, t):
        b,n,c,h,w = x.shape
        mask = torch.zeros_like(x)
        for count, i in enumerate(t):
            for j in range(-self.k,self.k+1):
                mask[count,i+j] = 1
        return x[mask==1].reshape(b,self.k*2+1,c,h,w)
    
    def stack_on_axis(self,x):
        return torch.cat([x[:,i] for i in range(self.k*2+1)], dim=1)
    
    def time_evolution(self, x, t, mu, sigma):
            
        W_t = torch.zeros_like(x)
        for count, i in enumerate(t):
            for j in range(i):
                W_t[count] += torch.randn_like(x)[0]
        prop = torch.exp((mu + 0.5*sigma**2) + sigma*W_t) #Propagator
        assert x.shape == prop.shape, f'Input shape {x.shape} must be equal to the propagator shape {prop.shape}'
        return x*prop

#         print(f"x start mean: {x.mean().item()} std:{x.std().item()}")


#         x_mean = x*torch.exp(mu)
#         x_var = torch.sqrt(x**2 * torch.exp(2*mu)*(torch.exp(sigma**2)-1))
#         x =  x_mean + x_var*torch.randn_like(x)
        
        
#         print(f"x after mean: {x.mean().item()} std:{x.std().item()}")
#         print(f"x_mean mean: {x_mean.mean().item()} std:{x_mean.std().item()}")
#         print(f"x_std start mean: {x_var.mean().item()} std:{x_var.std().item()}")
#         print(f"x**2 mean: {(x**2).mean().item()} std:{(x**2).std().item()}")
#         print(f"exp(2*mu) mean: {torch.exp(2*mu).mean().item()} std:{torch.exp(2*mu).std().item()}")
#         print(f"exp(sigma**2) mean: {torch.exp(sigma**2).mean().item()} std:{torch.exp(sigma**2).std().item()}")
#         print(f"mu start mean: {mu.mean().item()} std:{mu.std().item()}")
#         print(f"sigma start mean: {sigma.mean().item()} std:{sigma.std().item()}\n")
        
#         return x


#     def time_evolution(self, x, err):
# #         return (x-err)+ err*torch.randn_like(x)
#         return x-err

    def p_losses(self, x_start, t, y=None):
        x_start = self.sub_timeseries(x_start, t)
        
        gt_mu = torch.mean(x_start,dim=1)
        gt_sigma = torch.clamp(torch.std(x_start,dim=1),0.0000001)
        

        err = self.model(x_start[:,self.k-1], t)
        p_mu=err[:,:self.channels]
        p_sigma=torch.clamp(err[:,self.channels:],0.0000001)
        
#         print(p_mu.min(), p_mu.max(), gt_mu.min(), gt_mu.max())
        
        x_t = self.time_evolution(x_start[:,self.k-1], t, p_mu, p_sigma)

        if self.loss_type == 'l1':
            loss = (x_t - x_start[:,self.k]).abs().mean() 
        elif self.loss_type == 'l2':
            loss = F.mse_loss(x_t, x_start[:,self.k])

#         loss = F.mse_loss(gt_mu-p_mu) + F.mse_loss(gt_sigma, p_sigma)

#         loss = F.mse_loss(x_start[:,k]-x_start[:,k-1], err)
#         loss = F.mse_loss(mu + sigma*torch.randn_like(x_start[:,0]),torch.randn_like(x_start[:,0]))
#         loss += F.mse_loss(x_t, x_start[:,k])
        
#         loss = F.mse_loss((x_start[:,k]-x_start[:,k-1]), mu+sigma*torch.randn_like(x_start[:,k]))
    
        p = torch.distributions.normal.Normal(p_mu, p_sigma)
        q = torch.distributions.normal.Normal(gt_mu, gt_sigma)
#         q = torch.distributions.normal.Normal(torch.zeros_like(p_mu), torch.ones_like(p_mu))
        
        loss += torch.distributions.kl.kl_divergence(p, q).mean()

        return loss

    def forward(self, x, y=None, *args, **kwargs):
        x = x if len(x.shape) == 5 else x[:,:,None]
        b, n, c, h, w, device, img_size, = *x.shape, x.device, self.image_size
        assert h == img_size and w == img_size, f'height {h} and width {w} of image must be {img_size}'
        t = torch.randint(self.k, self.num_timesteps-self.k-1, (b,), device=device).long()
        return self.p_losses(x, t, y, *args, **kwargs)

# trainer class

class TrainerGBM(object):
    def __init__(
        self,
        model,
        train_loader,
        valid_loader,
        train_batch_size = 32,
        train_lr = 2e-5,
        train_num_steps = 100000,
        gradient_accumulate_every = 2,
        save_and_sample_every = 1000,
        results_folder = './logs',
        device = 'cuda'
    ):
        super().__init__()
        
        self.model = model
        
        self.save_and_sample_every = save_and_sample_every
        self.save_loss_every = 50

        self.batch_size = train_batch_size
        self.image_size = model.image_size
        self.gradient_accumulate_every = gradient_accumulate_every
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

    def save_with_1Dlabels(self, milestone, y, mode):
        writing_mode = 'w' if milestone == 1 else 'a+'
        with open(str(self.results_folder / f'labels-{mode}.txt'), writing_mode) as file:
            file.write(f"sample-{milestone}: ")
            for element in y:
                file.write(str(element.item()) + " ")
            file.write("\n")

    def save_with_2Dlabels(self, milestone, imgs, batches, mode, var_type):
        
        imgs_stacked = list(map(lambda n: imgs[:n], batches))
        imgs_stacked = torch.cat(imgs_stacked, dim=0)
        self.save_grid(imgs_stacked, str(self.results_folder / f'{milestone:03d}-{mode}-{var_type}.png'))

    def save_grid(self, images, file_name, nrow=5):
                
        grid = utils.make_grid(images, nrow=nrow)
        plt.figure()
        if images.shape[1] == 1:
            plt.imshow(grid[0].cpu().detach())
        else:
            plt.imshow(grid.permute((1,2,0)).cpu().detach())
        plt.savefig(file_name)

    def load(self):
        data = torch.load(str(self.results_folder / f'model.pt'))

        self.step = data['step']
        self.model.load_state_dict(data['model'])

    def train(self):

        avg_loss = [0,]

        while self.step < self.train_num_steps:

            for _, imgs in self.train_loader:
                x=imgs.to(self.device)
                
                self.opt.zero_grad()
                
                loss = self.model(x)
                loss.backward()
                
                if self.step != 0 and self.step % self.save_loss_every == 0:
                    avg_loss[-1] /= self.save_loss_every
                    print(f'Step: {self.step} - Loss: {avg_loss[-1]:.6f}')
                    avg_loss.append(0)

                avg_loss[-1] += loss.item()

                self.opt.step()

                if self.step != 0 and self.step % self.save_and_sample_every == 0:
                    _, imgs = next(iter(self.valid_loader))
                    x_val=imgs[:,-1].to(self.device)
                    x_val = x_val if len(x_val.shape) == 4 else x_val[:,None]
                    milestone = self.step // self.save_and_sample_every
                    n_images_to_sample = 25 
                    batches = num_to_groups(n_images_to_sample, self.batch_size) 
                    all_images_list = list(map(lambda n: self.model.sample(x_val[:n], batch_size=n), batches))    
                    all_images = torch.cat(all_images_list, dim=0)
                    # all_images = (all_images + 1.)*0.5
                    self.save_grid(all_images, str(self.results_folder / f'{milestone:03d}-train-pred.png'), nrow = 5)
                    self.save(avg_loss)
                    self.save_with_2Dlabels(milestone, imgs[:,-1][:,None], batches, mode='train', var_type='step_0')
                    self.save_with_2Dlabels(milestone, imgs[:,0][:,None], batches, mode='train', var_type='step_target')

                self.step += 1

        print('training completed')

    def test(self):
        _, imgs = next(iter(self.valid_loader))
        x=imgs[:,-1].to(self.device)
        
        milestone = self.step // self.save_and_sample_every
        n_images_to_sample = 25 
        batches = num_to_groups(n_images_to_sample, self.batch_size) 
        
        #Save Output
        all_images_list = list(map(lambda n: self.model.sample(x[:n], batch_size=n), batches))
        all_images = torch.cat(all_images_list, dim=0)
        # all_images = (all_images + 1) * 0.5
        self.save_grid(all_images, str(self.results_folder / f'{milestone:03d}-test-pred.png'), nrow = 5)

        self.save_with_2Dlabels(milestone, imgs[:,-1][:,None], batches, mode='test', var_type='step_0')
        self.save_with_2Dlabels(milestone, imgs[:,0][:,None], batches, mode='test', var_type='step_target')
        self.step += 1
            
            
    """TODO: Write GIFs and Entropy for GBM"""
    def GIFs(self, gif_type = 'sampling'):
        """Generate GIFs"""
        
        assert gif_type in ('sampling','histo'), 'gif_type should be one of (sampling,histo)'
        
        model = self.ema_model
        
        imgs, labels = next(iter(self.valid_loader))
        y = None if model.mode == 'unc' else labels.to(model.device)
        y = y[:25]
            
        shape = (25, model.channels, model.image_size, model.image_size)
        
        print(f'Generating {gif_type} GIF')
            
        b,_,h,w = shape
        img = torch.randn(shape, device=model.device)
        
        with imageio.get_writer(str(self.results_folder / f'{gif_type}.gif'), mode='I',fps=30) as writer:
            for i in tqdm(reversed(range(0, model.num_timesteps)), desc='sampling loop time step', total=model.num_timesteps):

                if gif_type == 'sampling':
                    grid = np.array(utils.make_grid(img, nrow=5).permute((1,2,0)).cpu().detach())
    #                 to_save = np.array(img[0].squeeze().detach().cpu()).copy()
                    writer.append_data((np.clip(grid*255, 0, 255).astype(np.uint8)))

                elif gif_type == 'histo':
                    # https://stackoverflow.com/questions/5320677/how-to-normalize-a-histogram-in-matlab
                    # Remark: bin_width < 1 --> p(x) can be higher than 1. The important thing is that the Area is 1.
                    # The Probability to have a value x is equal to p(x)*bin_width
                    hist,bin_edges = np.histogram(np.array(img.flatten().cpu().detach()), bins=30, range=(-2.0,2.0))
                    dx = bin_edges[1]-bin_edges[0]
                    plt.figure()
                    plt.scatter((bin_edges[:-1]+bin_edges[1:])/2, hist/(dx*hist.sum())) 
    #                 print(dx*(hist/(dx*hist.sum())).sum()) # Check Area = 1
                    plt.ylim(0.,1.0)
                    plt.xlabel('x')
                    plt.ylabel('pdf(x)')
                    plt.savefig('./logs/test.png')
                    plt.close()
                    npimage = imageio.imread('./logs/test.png')
                    writer.append_data(npimage)
                    os.remove('./logs/test.png')
                if self.model_type == 'c':
                    if len(y.shape) == 1: # Labels 1D
                        y = model.label_reshaping(y, b, h, w, model.device)   
                    img = model.label_concatenate(img,y)
                img = model.p_sample(img, torch.full((b,), i, dtype=torch.long, device=model.device))   
                
    def entropy(self):
        
        model = self.ema_model
        
        imgs, labels = next(iter(self.valid_loader))
        y = None if model.mode == 'unc' else labels.to(model.device)
        y = y[:25]
            
        shape = (25, model.channels, model.image_size, model.image_size)
        
        b,_,h,w = shape
        img = torch.randn(shape, device=model.device)
        
        entropy = []
        for i in tqdm(reversed(range(0, model.num_timesteps)), desc='sampling loop time step', total=model.num_timesteps):
            
            hist,bin_edges = np.histogram(np.array(img.flatten().cpu().detach()), bins=30, range=(-2.0,2.0))
            dx = bin_edges[1]-bin_edges[0]
            hist = hist/(dx*hist.sum())
            entropy.append(-(hist*np.log(np.abs(hist))).sum())

            if self.model_type == 'c':
                if len(y.shape) == 1: # Labels 1D
                    y = model.label_reshaping(y, b, h, w, model.device)   
                img = model.label_concatenate(img,y)
            img = model.p_sample(img, torch.full((b,), i, dtype=torch.long, device=model.device))  
            
        plt.figure()
        plt.scatter(np.linspace(1,1000,1000), entropy)
        plt.xlabel('Step')
        plt.ylabel('Entropy')
        plt.savefig(str(self.results_folder / 'entropy.png'))
