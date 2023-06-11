import torch
import torch.nn as nn

from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_act

from functools import partial


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

class SignedSqrt(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.sqrt(torch.relu(x)) - torch.sqrt(torch.relu(-x))
        return x



register_act('swish', partial(SWISH, inplace=cfg.mem.inplace))
register_act('lrelu_03', partial(nn.LeakyReLU, negative_slope=0.3, inplace=cfg.mem.inplace))
register_act('lrelu_02', partial(nn.LeakyReLU, negative_slope=0.2, inplace=cfg.mem.inplace))
# Add Gaussian Error Linear Unit (GELU).
register_act('gelu', nn.GELU)
