# Conditional Encoder

import torch
from torch import nn
from collections import OrderedDict

class Encoder(nn.Module):
    def __init__(
        self,
        dim, #e.g. 64 (second layer dimension)
        channels = 3,
        image_size = 32,
        dim_mults = (1, 2, 4, 8)
    ):
        super().__init__()
        
        dims = [channels, *map(lambda m: dim * m, dim_mults)]
        dim_post = dims[1]

        # self.net = nn.ModuleList([])

        for ind, dim in enumerate(dims):
            dim_prev = channels
            self.convs.append(
                 nn.Sequential(
                    OrderedDict([
                        (f'conv{ind+1}', nn.Conv2d(dim_prev, dim_post,3)),
                        (f'relu{ind+1}', nn.ReLU()),
                        (f'batch_norm{ind+1}', nn.BatchNorm2d(dim_post)),
                        (f'downsample{ind+1}', nn.MaxPool2d(2)),
                    ])))
        self.dense = nn.Sequential(
                        OrderedDict([
                            (f'linear1', nn.Linear()),
                            (f'relu{ind+1}', nn.ReLU()),
                            (f'relu{ind+1}', nn.ReLU()),
                            (f'relu{ind+1}', nn.ReLU()),
                        ])
        )

    # def forward():
