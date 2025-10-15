# Wrapper for your FUSNet conv model
import torch.nn as nn
import torch

class FUSNetWrapper(nn.Module):
    def __init__(self, args):
        super().__init__()
        # import the original class here if kept in legacy dir
        from legacy.FUSNet import MultiChannelConvolutionModel
        self.model = MultiChannelConvolutionModel(args)

    def forward(self, x):
        return self.model(x)