import torch.nn as nn
import torch

class SCONetWrapper(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        from legacy.SCONet import DepthwiseConvModel
        self.model = DepthwiseConvModel(**kwargs)
    def forward(self, x):
        return self.model(x)