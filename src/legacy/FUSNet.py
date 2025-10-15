import torch
import torch.nn as nn

class MultiChannelConvolutionModel(nn.Module):
    def __init__(self, args):
        super().__init__()

        try: self.context = args.context
        except AttributeError: raise AttributeError("Required attribute 'context' not found in args object.")
        try: self.input_channels_num = len(args.input_mics_numbers)
        except AttributeError: raise AttributeError("Required attribute 'input_mics_numbers' not found in args object.")
        try: self.output_channels_num = len(args.output_mics_numbers)
        except AttributeError: raise AttributeError("Required attribute 'output_mics_numbers' not found in args object.")
        
        self.filter_length = int(2*self.context)+1 
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(
                in_channels=1,
                out_channels=self.output_channels_num,
                kernel_size=self.filter_length,
                bias=False
            )
            for _ in range(self.input_channels_num)
        ])
        
    def forward(self, x):
        
        outs = [conv(x[:, i:i+1, :]) for i, conv in enumerate(self.conv_layers)]
        out = torch.sum(torch.stack(outs, dim=0), dim=0)
        return out