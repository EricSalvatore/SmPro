import torch
import torch.nn as nn

"""
the class of the residual module
"""

class ResidualNet(nn.Module):
    def __init__(self, fn):
        super(ResidualNet, self).__init__()
        self.fn = fn

    def forward(self, x):
        return x + self.fn(x)