import torch
import torch.nn as nn

class DepthwiseConvModel(nn.Module):
    def __init__(self, args):
        super(DepthwiseConvModel, self).__init__()
        # self.stft_transform = T.Spectrogram(n_fft=n_fft, hop_length=win_length//2, win_length=win_length, power=None) 
        
        try: self.n_fft = args.n_fft
        except AttributeError: raise AttributeError("Required attribute 'n_fft' not found in args object.")
        try: self.input_channels_num = len(args.input_mics_numbers)
        except AttributeError: raise AttributeError("Required attribute 'input_mics_numbers' not found in args object.")
        try: self.output_channels_num = len(args.output_mics_numbers)
        except AttributeError: raise AttributeError("Required attribute 'output_mics_numbers' not found in args object.")
        
        self.win_length = self.n_fft    # 2**13
        self.channels = self.n_fft//2+1 # Set based on your data
        self.hop_length = self.win_length // 2
        self.n_output_channels = self.output_channels_num*2

        self.depthwise_conv = nn.Conv2d(
            in_channels= self.channels,
            out_channels= self.n_output_channels * self.channels,
            kernel_size=(2*self.input_channels_num, 1),
            stride=1,
            padding=0,
            groups=self.channels,
            bias=False    
        )


    def forward(self, audio_signal): # audio_signal shape: (batch_size, num_channels, time)
        # Compute STFT
        stft_result = torch.stft(audio_signal, n_fft=self.n_fft, win_length=self.n_fft, hop_length=self.hop_length, window=torch.hann_window(self.n_fft).to(audio_signal.device), return_complex=False)  # Shape: (batch_size, num_channels, freq, time, 2)
        stft_result = torch.cat((stft_result[:,:,:,:,0],stft_result[:,:,:,:,1]),dim=1) # Shape: (batch_size, num_channels*2, freq, time)
        stft_result = torch.swapaxes(stft_result, 1, 2) # Shape: (batch_size, freq, num_channels*2, time)
        output = self.depthwise_conv(stft_result) # Shape: (batch_size, out_channels*freq, 1, time)
        output = output.squeeze(2) # Shape: (batch_size, out_channels*freq, time)
        output = self.divide_and_stack(output, self.n_output_channels, axis=1) # Shape: (batch_size, freq, out_channels, time)
        output = torch.swapaxes(output, 1, 2)   # Shape: (batch_size, out_channels, freq, time)
        return output

    @staticmethod
    def divide_and_stack(tensor, num_splits, axis):
        divided_tensors = torch.split(tensor, num_splits, dim=axis)
        
        return torch.stack(divided_tensors, dim=1)