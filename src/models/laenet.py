# Wrapper for FrequencyLSTMNet
import torch.nn as nn
import torch

class LAENetWrapper(nn.Module):
    def __init__(self, args):
        super().__init__()
        from legacy.LAENet import FrequencyLSTMNet
        self.model = FrequencyLSTMNet(args)
        self.input_mic_numbers = args.input_mics_numbers - 1
    def get_output_stft(self, input_signal, retm_stft): # input_signal -> (batch_size, input_channels_num, time_steps)
        input_stft = self.model.compute_stft(input_signal)
        batch_size, freq, time_dim, _ = input_stft.shape
        input_stft = torch.view_as_real(input_stft).reshape(batch_size, freq, time_dim, -1)
        retm_stft = retm_stft.mean(axis=0).unsqueeze(0).repeat(batch_size, 1, 1, 1)
        output_stft = torch.einsum('bfnc,bftc->bftn', retm_stft, input_stft)
        output_stft = output_stft[:,:,:,0::2] + 1j*output_stft[:,:,:,1::2]
        output_stft = output_stft.permute(0, 3, 1, 2)  # (batch, channels, time, freq)
        return output_stft
    def forward(self, x):   # x -> batch, total_channels_num, time_steps
        retm_stft = self.model(x)
        input_signal = x[:,self.input_mic_numbers,:]
        return retm_stft, self.get_output_stft(input_signal, retm_stft)
    