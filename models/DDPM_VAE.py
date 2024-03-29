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

import mlflow

import pytorch_lightning as pl

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

# gaussian diffusion trainer class

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, a_min = 0, a_max = 0.999)

def image_hist2d(image: torch.Tensor, n_bins: int = 256, bandwidth: float = -1.,
                 centers: torch.Tensor = torch.tensor([]), return_pdf: bool = False):
    """Function that estimates the histogram of the input image(s).

    The calculation uses triangular kernel density estimation.

    Args:
        x: Input tensor to compute the histogram with shape
        :math:`(H, W)`, :math:`(C, H, W)` or :math:`(B, C, H, W)`.
        min: Lower end of the interval (inclusive).
        max: Upper end of the interval (inclusive). Ignored when
        :attr:`centers` is specified.
        n_bins: The number of histogram bins. Ignored when
        :attr:`centers` is specified.
        bandwidth: Smoothing factor. If not specified or equal to -1,
        bandwidth = (max - min) / n_bins.
        centers: Centers of the bins with shape :math:`(n_bins,)`.
        If not specified or empty, it is calculated as centers of
        equal width bins of [min, max] range.
        return_pdf: If True, also return probability densities for
        each bin.

    Returns:
        Computed histogram of shape :math:`(bins)`, :math:`(C, bins)`,
        :math:`(B, C, bins)`.
        Computed probability densities of shape :math:`(bins)`, :math:`(C, bins)`,
        :math:`(B, C, bins)`, if return_pdf is ``True``. Tensor of zeros with shape
        of the histogram otherwise.
    """
    v_min = image.min().item()
    v_max = image.max().item()
    
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input image type is not a torch.Tensor. Got {type(image)}.")

    if centers is not None and not isinstance(centers, torch.Tensor):
        raise TypeError(f"Bins' centers type is not a torch.Tensor. Got {type(centers)}.")

    if centers.numel() > 0 and centers.dim() != 1:
        raise ValueError(f"Bins' centers must be a torch.Tensor "
                         "of the shape (n_bins,). Got {values.shape}.")

    if not isinstance(n_bins, int):
        raise TypeError(f"Type of number of bins is not an int. Got {type(n_bins)}.")

    if bandwidth != -1 and not isinstance(bandwidth, float):
        raise TypeError(f"Bandwidth type is not a float. Got {type(bandwidth)}.")

    if not isinstance(return_pdf, bool):
        raise TypeError(f"Return_pdf type is not a bool. Got {type(return_pdf)}.")
        
    device = image.device

    if image.dim() == 4:
        batch_size, n_channels, height, width = image.size()
    elif image.dim() == 3:
        batch_size = 1
        n_channels, height, width = image.size()
    elif image.dim() == 2:
        height, width = image.size()
        batch_size, n_channels = 1, 1
    else:
        raise ValueError(f"Input values must be a of the shape BxCxHxW, "
                         f"CxHxW or HxW. Got {image.shape}.")

    if bandwidth == -1.:
        bandwidth = (v_max - v_min) / n_bins
    if centers.numel() == 0:
        centers = v_min + bandwidth * (torch.arange(n_bins, device=device).float() + 0.5)
    centers = centers.reshape(-1, 1, 1, 1, 1)
    u = abs(image.unsqueeze(0) - centers) / bandwidth
    mask = (u <= 1).float()
    hist = torch.sum(((1 - u) * mask), dim=(-2, -1)).permute(1, 2, 0)
    if return_pdf:
        normalization = torch.sum(hist, dim=-1).unsqueeze(0) + 1e-10
        pdf = hist / normalization
        return hist, pdf
    return hist#, torch.zeros_like(hist, dtype=hist.dtype, device=device)

class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        *,
        model_type = 'unc',
        image_size,
        channels = 3,
        timesteps = 1000,
        sample_every = 1,
        lam = 0.001,
        loss_type = 'l1',
        metrics_type = 'PSNR',
        betas = None,
        device='cuda:0'
    ):
        super().__init__()
        self.mode = model_type
        self.channels = channels
        self.image_size = image_size
        self.denoise_fn = denoise_fn.to(device)
        self.device=device

        self.sample_every = sample_every
        self.lam = lam
        self.bins = 256
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type
        self.metrics_type = metrics_type
        self.loss_simple = nn.MSELoss()
        self.loss_vlb = torch.nn.KLDivLoss()
        self.histogram = image_hist2d
        self.metrics = self._set_metrics()

        self.betas = betas
        
        self.set_parameters(state='train')


    def set_parameters(self,state='train'):
        
        to_torch = partial(torch.tensor, dtype=torch.float32, device=self.device)
        
#         if exists(self.betas):
#             betas = self.betas.detach().cpu().numpy() if isinstance(self.betas, torch.Tensor) else self.betas
#         else:
#             betas = cosine_beta_schedule(self.num_timesteps)

        betas = cosine_beta_schedule(self.num_timesteps)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        self.betas = to_torch(betas)
        self.alphas_cumprod = to_torch(alphas_cumprod)
        self.alphas_cumprod_prev = to_torch(alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = to_torch(np.sqrt(alphas_cumprod))
        self.sqrt_one_minus_alphas_cumprod = to_torch(np.sqrt(1. - alphas_cumprod))
        self.log_one_minus_alphas_cumprod = to_torch(np.log(1. - alphas_cumprod))
        self.sqrt_recip_alphas_cumprod = to_torch(np.sqrt(1. / alphas_cumprod))
        self.sqrt_recipm1_alphas_cumprod = to_torch(np.sqrt(1. / alphas_cumprod - 1))

        if state == 'test':
            # Fast sampling
            alphas =  alphas[::self.sample_every]
            alphas_cumprod = alphas_cumprod[::self.sample_every]
            alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
            betas = betas[::self.sample_every]
#             betas = 1. - alphas_cumprod/alphas_cumprod_prev
#             self.betas = to_torch(betas)
        
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.posterior_variance = to_torch(posterior_variance)
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_log_variance_clipped = to_torch(np.log(np.maximum(posterior_variance, 1e-20)))
        self.posterior_mean_coef1 =  to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.posterior_mean_coef2 = to_torch((1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))
        
        
    def _set_loss(self):
        if self.loss_type == 'l1':
            loss = lambda x,y: (x - y).abs().mean()
        elif self.loss_type == 'l2':
            loss = nn.MSELoss()
#             F.mse_loss(noise, x_recon)
        else:
            raise NotImplementedError()
        return loss
    
    def _set_metrics(self):
        if self.metrics_type == 'PSNR':
            metrics = lambda x,y: 20*torch.log10(x.max()) - 10*torch.log10(((x-y)**2).mean())
            
        return metrics
    
    def to_histo(self, t):
        assert t.ndim == 4, f'Shape should be (B,C,H,W) instead of tensor.shape'
        return self.histogram(t,n_bins=self.bins)#.reshape(-1,self.bins)

    def set_min_log(self,t):
        return torch.maximum(t,torch.ones_like((t))*1e-20).to(self.device)   

    def split_diffusion_output(self,t):
        assert t.ndim == 4, f'Shape should be (B,C,H,W) instead of tensor.shape'
        _,channels,*_ = t.shape
        assert channels%2 == 0, 'Output channels should be even'
        return t[:,:channels//2], t[:,channels//2:]

    def label_reshaping(self, y, b, h, w, device):
        # y = torch.tensor(y)
        assert len(y.shape) == 1, 'labels array is 1D'
        assert torch.is_tensor(y), 'labels array should be a pytorch tensor'
        labels = torch.ones((b,1,h,w)).to(device)
        return torch.einsum('ijkl,i->ijkl', labels, y)

    def label_concatenate(self, x,y):
        return torch.cat([x,y],dim=1)

    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool):
        noise = self.denoise_fn(x, t)
        x_recon = self.predict_start_from_noise(x[:,:self.channels], t=t, noise=noise)

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x[:,:self.channels], t=t)
        return model_mean, posterior_variance, posterior_log_variance

    def p_sample(self, x, t, clip_denoised=True, repeat_noise=False):
        b, *_ = x.shape
        model_mean, model_variance, model_log_variance = self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised)
        noise = noise_like(x[:,:self.channels].shape, self.device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * model_variance * noise

    @torch.no_grad()
    def p_sample_loop(self, y, shape):
        self.set_parameters(state='test')
        b,_,h,w = shape
        img = torch.randn(shape, device=self.device)
        for i in reversed(range(0, self.num_timesteps//self.sample_every)):
            if self.mode == 'c':
                if len(y.shape) == 1: # Labels 1D
                    y = self.label_reshaping(y, b, h, w, self.device)   
                img = self.label_concatenate(img,y)
            img = self.p_sample(img, torch.full((b,), i, dtype=torch.long, device=self.device))   
        self.set_parameters(state='train')
        return img

    @torch.no_grad()
    def sample(self, y=None, batch_size = 16):
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop(y, (batch_size, channels, image_size, image_size))

    @torch.no_grad()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in tqdm(reversed(range(0, t)), desc='interpolation sample time step', total=t):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))

        return img

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start)).to(x_start.device)

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, y=None, noise = None):
        b, c, h, w = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start)).to(x_start.device)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        if self.mode == 'c':
            x_noisy = self.label_concatenate(x_noisy, y)
        pred_noise = self.denoise_fn(x_noisy, t)
        
#         x_recon = self.predict_start_from_noise(x_noisy[:,:self.channels], t=t, noise=pred_noise)
#         x_recon.clamp_(-1., 1.)
#         x_start.clamp_(-1., 1.)
            
        return self.loss_simple(noise,pred_noise)# + 0.1*self.loss_simple(x_start,x_recon)

    def forward(self, x, y=None, *args, **kwargs):
        b, c, h, w, device, img_size, = *x.shape, x.device, self.image_size
        assert h == img_size and w == img_size, f'height {h} and width {w} of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        if self.mode == 'c':
            assert torch.is_tensor(y) and y.device == device
            if len(y.shape) == 1: # Labels 1D
                y = self.label_reshaping(y, b, h, w, device)
        return self.p_losses(x, t, y, *args, **kwargs)

# trainer class

class LitModelDDPM(pl.LightningModule):
    def __init__(
        self,
        diffusion_model,
        model_type = 'unc',
        batch_size = 32,
        lr = 2e-5,
        save_loss_every = 50
    ):
        super(LitModelDDPM, self).__init__()
        
        self.model = diffusion_model        
        self.model_type = model_type

        self.batch_size = batch_size
        self.lr = lr
        self.save_loss_every = save_loss_every
        self.lower_loss=1000000
        
    def forward(self, x):        
        return self.model.sample(x)
    
    def set_batch(self,batch):
        x,y = batch
        return x, None if self.model_type == 'unc' else y
        
    def training_step(self, batch, batch_nb):
        x, y = self.set_batch(batch)
        
        loss = self.model(x,y)
        
        if self.global_step % self.save_loss_every == 0 and self.global_step != 0:
            self.logger.experiment.log_metric(run_id = self.logger.run_id,
                                              key = 'loss', 
                                              value = loss.item(),
                                              step = self.global_step)
            
            if loss < self.lower_loss:
                torch.save(self.model.denoise_fn.state_dict(), f'/nfs/conditionalDDPM/tmp/model_{self.global_step}.pt')
                torch.save(self.model.denoise_fn.state_dict(), f'/nfs/conditionalDDPM/tmp/model_best.pt')
                self.logger.experiment.log_artifact(run_id = self.logger.run_id,
                                                            local_path=f'/nfs/conditionalDDPM/tmp/model_best.pt', 
                                                            artifact_path=None)
                self.lower_loss = loss
                    
        return loss
    
    @torch.no_grad()
    def test_step(self, batch, batch_nb):
        x, y = self.set_batch(batch)
        test_batch = 4
        n_row = int(test_batch**0.5)

        all_images = self.model.sample(y if y == None else y[:test_batch], batch_size=test_batch)  
        if y.ndim > 1:
            metrics_pred = self.model.metrics(all_images, y[:test_batch])
            metrics_target = self.model.metrics(x[:test_batch], y[:test_batch])

            self.logger.experiment.log_metric(run_id = self.logger.run_id,
                                              key = self.model.metrics_type+'-pred', 
                                              value = metrics_pred.item(),
                                              step = self.global_step)

            self.logger.experiment.log_metric(run_id = self.logger.run_id,
                                              key = self.model.metrics_type+'-target', 
                                              value = metrics_target.item(),
                                              step = self.global_step)

        self.save_with_2Dlabels(all_images, mode='test', var_type='pred', n_row=n_row)

        if self.model_type == 'c':
            if y.ndim == 1:
                self.save_with_1Dlabels(y, mode='test', n_row=n_row) 
                os.remove(f'/nfs/conditionalDDPM/tmp/labels-test.txt')
            else:
                self.save_with_2Dlabels(x, mode='test', var_type='target', n_row=n_row)
                self.save_with_2Dlabels(y if y.shape[1]!= 2 else y[:,0][:,None], 
                                        mode='test', var_type='input', n_row=n_row)
                
#         torch.save(self.model.state_dict(), f'/nfs/conditionalDDPM/tmp/model.pt')
#         self.logger.experiment.log_artifact(run_id = self.logger.run_id,
#                                                     local_path=f'/nfs/conditionalDDPM/tmp/model.pt', 
#                                                     artifact_path=None)
                    
        """TODO: Add also GIF!!"""

    def configure_optimizers(self):
#         optimizers = [torch.optim.Adam(self.model.parameters(), lr=self.lr),]
#         schedulers = [{
#             "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizers[0], patience=5),
#             "monitor": 'loss',
#             "frequency": 1
#                         },]
#         return optimizers, schedulers):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)
           
    def save_grid(self, images, file_name, nrow=3):
                
            
        vmin=images.min()
        vmax=images.max()
        
        grid = utils.make_grid(images, nrow=nrow, normalize=False)
        plt.figure()
        if images.shape[1] == 1:
            plt.imshow(grid[0].cpu().detach())
        else:
            plt.imshow(grid.permute((1,2,0)).cpu().detach())
        plt.colorbar()
        plt.savefig(file_name)
        
    def save_with_1Dlabels(self, y, mode, n_row=3):
        writing_mode = 'w' if not os.path.exists(f'/nfs/conditionalDDPM/tmp/labels-{mode}.txt') else 'a+'
        with open(f'/nfs/conditionalDDPM/tmp/labels-{mode}.txt', writing_mode) as file:
            for element in y[:n_row**2]:
                file.write(str(element.item()) + " ")
            file.write("\n")
        
        self.logger.experiment.log_artifact(run_id = self.logger.run_id,
                                            local_path=f'/nfs/conditionalDDPM/tmp/labels-{mode}.txt', 
                                            artifact_path='training')

    def save_with_2Dlabels(self, imgs, mode, var_type, n_row=3):
        imgs_stacked = imgs[:n_row**2]
        self.save_grid(imgs_stacked, f'/nfs/conditionalDDPM/tmp/{mode}-{var_type}.png', nrow = n_row)
        self.logger.experiment.log_artifact(run_id = self.logger.run_id,
                                            local_path=f'/nfs/conditionalDDPM/tmp/{mode}-{var_type}.png', 
                                            artifact_path='training')
            
    def GIFs(self, val_batch, gif_type = 'sampling'):
        """Generate GIFs"""
        
        assert gif_type in ('sampling','histo'), 'gif_type should be one of (sampling,histo)'
        
        x, y = self.set_batch(batch)
                            
        y = y[:9]
            
        shape = (9, model.channels, model.image_size, model.image_size)
        
        print(f'Generating {gif_type} GIF')
            
        b,_,h,w = shape
        img = torch.randn(shape, device=model.device)
        
        with imageio.get_writer(str(f'/nfs/conditionalDDPM/tmp/{gif_type}.gif'), mode='I',fps=30) as writer:
            for i in tqdm(reversed(range(0, model.num_timesteps)), desc='sampling loop time step', total=model.num_timesteps):

                if gif_type == 'sampling':
                    grid = np.array(utils.make_grid(img, nrow=3).permute((1,2,0)).cpu().detach())
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
                    plt.savefig('./tmp/test.png')
                    plt.close()
                    npimage = imageio.imread('./logs/test.png')
                    writer.append_data(npimage)
                    os.remove('./tmp/test.png')
                if self.model_type == 'c':
                    if len(y.shape) == 1: # Labels 1D
                        y = self.model.label_reshaping(y, b, h, w, model.device)   
                    img = self.model.label_concatenate(img,y)
                img = self.model.p_sample(img, torch.full((b,), i, dtype=torch.long, device=model.device))   