# Wrapper for FrequencyLSTMNet
import torch.nn as nn
import torch

class LAENetWrapper(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        from legacy.LAENet import FrequencyLSTMNet
        self.model = FrequencyLSTMNet(**kwargs)
    def forward(self, x):
        return self.model(x)