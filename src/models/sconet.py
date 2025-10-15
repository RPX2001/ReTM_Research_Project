import torch.nn as nn
import torch

class SCONetWrapper(nn.Module):
    def __init__(self, args):
        super().__init__()
        from legacy.SCONet import DepthwiseConvModel
        self.model = DepthwiseConvModel(args)
    def forward(self, x): # x -> (batch_size, input_channels_num, time_steps)
        return self.model(x)