import torch
import torch.nn as nn
from graphgym.config import cfg
from graphgym.register import register_act


class SWISH(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        if self.inplace:
            x.mul_(torch.sigmoid(x))
            return x
        else:
            return x * torch.sigmoid(x)

class IDENTITY(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return x


register_act('swishs', SWISH(inplace=cfg.mem.inplace))
register_act('lrelu_03', nn.LeakyReLU(negative_slope=0.3, inplace=cfg.mem.inplace))

register_act('identity', IDENTITY())
register_act('sigmoid', nn.Sigmoid())
register_act('tanh', nn.Tanh())


