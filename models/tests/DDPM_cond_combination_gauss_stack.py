"""Copyright (c) 2020 Phil Wang"""

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
        self.dim = dim
        self.out_dim = out_dim
        self.dim_mults = dim_mults
        self.groups = groups
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

class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        *,
        model_type = 'unc',
        image_size,
        channels = 3,
        timesteps = 1000,
        loss_type = 'l1',
        betas = None,
        device='cuda'
    ):
        super().__init__()
        self.mode = model_type
        self.channels = channels
        self.image_size = image_size
        self.denoise_fn = denoise_fn
        if self.mode == 'c':            
            self.noise_gen = Unet(
                                dim = self.denoise_fn.dim,
                                dim_mults = self.denoise_fn.dim_mults,
                                channels = self.denoise_fn.channels-1,
                                out_dim = (self.denoise_fn.channels-1)*2, 
                                with_time_emb = False)
            self.noise_gen = self.noise_gen.to(device)

        self.device=device

        if exists(betas):
            betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        else:
            betas = cosine_beta_schedule(timesteps)

        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        to_torch = partial(torch.tensor, dtype=torch.float32, device=self.device)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))


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
        x_recon = self.predict_start_from_noise(x[:,:self.channels], t=t, noise=self.denoise_fn(x, t))

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x[:,:self.channels], t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, repeat_noise=False):
        b, *_ = x.shape
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised)
        noise = noise_like(x[:,:self.channels].shape, self.device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, y, shape):
        b,c,h,w = shape
        
        if self.mode == 'c':
            output = self.noise_gen(y, None)
            mu, log_var = output[:,:c], output[:,c:]
            img = torch.normal(mu, log_var.exp()).to(self.device)
#             img = self.noise_gen(y,None)
        else:
            img = torch.randn(shape, device=self.device)
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            if self.mode == 'c':
                if len(y.shape) == 1: # Labels 1D
                    y = self.label_reshaping(y, b, h, w, self.device)   
                img = self.label_concatenate(img,y)
            img = self.p_sample(img, torch.full((b,), i, dtype=torch.long, device=self.device))            
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
        if self.mode == 'c':
            output = self.noise_gen(y, None)
            mu, log_var = output[:,:c], output[:,c:]
            noise = torch.normal(mu, log_var.exp()).to(x_start.device)
        else:
            noise = default(noise, lambda: torch.randn_like(x_start)).to(x_start.device)
        
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        
        if self.mode == 'c':
            x_noisy = self.label_concatenate(x_noisy, y)
            
        x_recon = self.denoise_fn(x_noisy, t)

        if self.loss_type == 'l1':
            loss = (noise - x_recon).abs().mean()
            if self.mode == 'c':
                loss += (noise - default(noise, lambda: torch.randn_like(x_start)).to(x_start.device)).abs().mean()
        elif self.loss_type == 'l2':
            loss = F.mse_loss(noise, x_recon)
#             print('loss1',loss)
            if self.mode == 'c':
#                 loss += F.mse_loss(noise, torch.randn_like(x_start, device=x_start.device))
#                 loss_noise = torch.nn.KLDivLoss()(noise, torch.randn_like(x_start, device=x_start.device))
                loss_noise = -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp())
                loss += loss_noise
#                 print('loss2', loss_noise)
        else:
            raise NotImplementedError()
        
        return loss

    def forward(self, x, y=None, *args, **kwargs):
        b, c, h, w, device, img_size, = *x.shape, x.device, self.image_size
        assert h == img_size and w == img_size, f'height {h} and width {w} of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        if self.mode == 'c':
            assert torch.is_tensor(y) and y.device == device
            if len(y.shape) == 1: # Labels 1D
                y = self.label_reshaping(y, b, h, w, device)
        return self.p_losses(x, t, y, *args, **kwargs)


# dataset classes

class Dataset(data.Dataset):
    def __init__(self, folder, image_size, exts = ['jpg', 'jpeg', 'png']):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1)
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)

# trainer class

class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        train_loader,
        valid_loader,
        *,
        model_type = 'unc',
        ema_decay = 0.995,
        train_batch_size = 32,
        train_lr = 2e-5,
        train_num_steps = 100000,
        gradient_accumulate_every = 2,
        fp16 = False,
        step_start_ema = 2000,
        update_ema_every = 10,
        save_and_sample_every = 1000,
        results_folder = './results',
        device = 'cuda'
    ):
        super().__init__()
        self.model = diffusion_model
        self.model_type = model_type
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every
        self.save_loss_every = 50

        self.batch_size = train_batch_size
        self.image_size = diffusion_model.image_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps

        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.opt = Adam(diffusion_model.parameters(), lr=train_lr)

        self.step = 0

        self.device = device

        assert not fp16 or fp16 and APEX_AVAILABLE, 'Apex must be installed in order for mixed precision training to be turned on'

        self.fp16 = fp16
        if fp16:
            (self.model, self.ema_model), self.opt = amp.initialize([self.model, self.ema_model], self.opt, opt_level='O1')

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True, parents=True)

        self.reset_parameters()

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def save(self, loss):
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict(),
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
        self.ema_model.load_state_dict(data['ema'])

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
                    milestone = self.step // self.save_and_sample_every
                    n_images_to_sample = 25 
                    batches = num_to_groups(n_images_to_sample, self.batch_size) 
                    all_images_list = list(map(lambda n: self.ema_model.sample(y if y == None else y[:n], batch_size=n), batches))    
                    all_images = torch.cat(all_images_list, dim=0)
                    # all_images = (all_images + 1.)*0.5
                    self.save_grid(all_images, str(self.results_folder / f'{milestone:03d}-train-pred.png'), nrow = 5)
                    self.save(avg_loss)
                    if self.model_type == 'c':
                        if len(y.shape) == 1:
                            self.save_with_1Dlabels(milestone, y, mode='train') 
                        else:
                            self.save_with_2Dlabels(milestone, x, batches, mode='train', var_type='original')
                            self.save_with_2Dlabels(milestone, y, batches, mode='train', var_type='condition')
                            mu_logvar_list = list(map(lambda n: self.ema_model.noise_gen(y[:n], None), batches)) 
                            mu_logvar = torch.cat(mu_logvar_list, dim=0)
                            b,c,h,w = mu_logvar.shape
                            mu, log_var = mu_logvar[:,:int(c//2)], mu_logvar[:,int(c//2):]
                            noise = torch.normal(mu,log_var.exp())
                            self.save_grid(noise, str(self.results_folder / f'{milestone:03d}-train-noise.png'), nrow = 5)

                self.step += 1

        print('training completed')

    def test(self):
        imgs, labels = next(iter(self.train_loader))
        x=imgs.to(self.device)
        if self.model_type == 'unc':
            y=None
        elif self.model_type == 'c':
            y=labels.to(self.device)

        milestone = self.step // self.save_and_sample_every
        n_images_to_sample = 25 
        batches = num_to_groups(n_images_to_sample, self.batch_size) 
        
        #Save Output
        all_images_list = list(map(lambda n: self.ema_model.sample(y if y == None else y[:n], batch_size=n), batches))
        all_images = torch.cat(all_images_list, dim=0)
        # all_images = (all_images + 1) * 0.5
        self.save_grid(all_images, str(self.results_folder / f'{milestone:03d}-test-pred.png'), nrow = 5)

        if self.model_type == 'c':
            if len(y.shape) == 1:
                self.save_with_1Dlabels(milestone, y, mode='test') 
            else:
                self.save_with_2Dlabels(milestone, x, batches, mode='test', var_type='original')
                self.save_with_2Dlabels(milestone, y, batches, mode='test', var_type='condition')
            self.step += 1