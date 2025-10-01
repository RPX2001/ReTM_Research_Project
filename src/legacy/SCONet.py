import torch
import torch.nn as nn

win_length = 2**13
n_fft = win_length
hop_length = win_length // 2

class DepthwiseConvModel(nn.Module):
    def __init__(self, channels, n_output_channels):
        super(DepthwiseConvModel, self).__init__()
        # self.stft_transform = T.Spectrogram(n_fft=n_fft, hop_length=win_length//2, win_length=win_length, power=None) 
        self.depthwise_conv = nn.Conv2d(
            in_channels=channels,
            out_channels=n_output_channels * channels,
            kernel_size=(2*4, 1),
            stride=1,
            padding=0,
            groups=channels,
            bias=False    
        )
        self.n_fft = n_fft
        self.hop_length = win_length // 2
        self.n_output_channels = n_output_channels


    def forward(self, audio_signal):
        # Compute STFT
        stft_result = torch.stft(audio_signal, n_fft=self.n_fft, win_length=self.n_fft, hop_length=self.hop_length, window=torch.hann_window(self.n_fft).to(audio_signal.device), return_complex=False)  # Shape: (num_channels, freq, time)
        
        stft_result = torch.cat((stft_result[:,:,:,0],stft_result[:,:,:,1]),dim=0)

        stft_result = torch.swapaxes(stft_result, 0, 1)

        output = self.depthwise_conv(stft_result)

        output = output.squeeze()
        
        output = self.divide_and_stack(output, self.n_output_channels, axis=0)
        
        output = torch.swapaxes(output, 0, 1)
        
 
        return output

    @staticmethod
    def divide_and_stack(tensor, num_splits, axis):
        divided_tensors = torch.split(tensor, num_splits, dim=axis)
        
        return torch.stack(divided_tensors, dim=0)