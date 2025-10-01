import torch
import torch.nn as nn

class MultiChannelConvolutionModel(nn.Module):
    def __init__(self, filter_length):
        super(MultiChannelConvolutionModel, self).__init__()
        
        # Define 4 convolutional filters for the 4 channels of audio
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=3, kernel_size=filter_length, bias=False)
        self.conv2 = nn.Conv1d(in_channels=1, out_channels=3, kernel_size=filter_length, bias=False)
        self.conv3 = nn.Conv1d(in_channels=1, out_channels=3, kernel_size=filter_length, bias=False)
        self.conv4 = nn.Conv1d(in_channels=1, out_channels=3, kernel_size=filter_length, bias=False)
        
    def forward(self, x):
        # Apply the convolution filters on each channel (dim=1)
        conv1_out = self.conv1(x[:, 0:1, :])  # Only take the 1st channel for convolution
        conv2_out = self.conv2(x[:, 1:2, :])  # Only take the 2nd channel for convolution
        conv3_out = self.conv3(x[:, 2:3, :])  # Only take the 3rd channel for convolution
        conv4_out = self.conv4(x[:, 3:4, :])  # Only take the 4th channel for convolution
        
        # Average the outputs of all channels
        out = (conv1_out + conv2_out + conv3_out + conv4_out) 
        
        # Return the averaged result
        return out