import torch
import torch.nn as nn
import torch.nn.functional as F


class FrequencyLSTMNet(nn.Module):
    def __init__(self, args):
        super(FrequencyLSTMNet, self).__init__()

        try: self.n_fft = args.n_fft
        except AttributeError: raise AttributeError("Required attribute 'n_fft' not found in args object.")
        try: self.lstm_layers = args.lstm_layers
        except AttributeError: raise AttributeError("Required attribute 'lstm_layers' not found in args object.")
        try: self.input_channels_num = len(args.input_mics_numbers)
        except AttributeError: raise AttributeError("Required attribute 'input_mics_numbers' not found in args object.")
        try: self.output_channels_num = len(args.output_mics_numbers)
        except AttributeError: raise AttributeError("Required attribute 'output_mics_numbers' not found in args object.")

        self.channel_num = self.input_channels_num + self.output_channels_num
        self.num_frequencies = self.n_fft//2 + 1    # In the original code, this was 4097, which is 2**13/2 + 1 where n_fft = 2**13
        self.lstm_hidden_size = self.channel_num
        self.hop_length = self.n_fft//2             # 2**13//2
        self.hann_window = torch.hann_window(self.n_fft)
        
        self.layer_norms = nn.ModuleList([nn.LayerNorm(4*self.lstm_hidden_size) for _ in range(self.num_frequencies)])
        self.lstm  = nn.LSTM(input_size = 2*self.input_channels_num, hidden_size = 2*self.lstm_hidden_size, num_layers=self.lstm_layers, batch_first=True, bidirectional=True)
        
        # Fully connected layers
        self.fc1 = nn.Linear(4*self.lstm_hidden_size, 8*(self.output_channels_num*self.input_channels_num))
        self.fc2 = nn.Linear(8*(self.output_channels_num*self.input_channels_num), 16*(self.output_channels_num*self.input_channels_num))
        self.fc3 = nn.Linear(16*(self.output_channels_num*self.input_channels_num), 4*(self.output_channels_num*self.input_channels_num))        
    
    def forward(self, x):
        
        batch_size, channel_num, time_steps = x.shape
        stft = self.compute_stft(x) # batch, freq, time, channels
        time_dim = stft.shape[2]    # Correct time dimension after STFT
        lstm_direct_outs = torch.zeros(batch_size, self.num_frequencies, 4 * self.lstm_hidden_size).to(x.device)  # (batch, freq, time, 2 * lstm_hidden_size)

        lstm_outputs = []
        for f in range(self.num_frequencies):
            lstm_input = torch.view_as_real(stft[:, f, :, :]).reshape(batch_size, time_dim, -1)  # (batch, STFT_time, (total_channels_num * 2))
            lstm_out, _ = self.lstm(lstm_input)
            lstm_out = self.layer_norms[f](lstm_out)
            
            lstm_out = lstm_out.mean(axis=1)
            lstm_direct_outs[:, f, :] = lstm_out  # (batch, freq, 2 * lstm_hidden_size)
            lstm_out = lstm_out.view(batch_size, -1)  # (batch, 2 * lstm_hidden_size))
            fcout = self.fc1(lstm_out)
            fcout = F.relu(fcout)
            fcout = self.fc2(fcout)
            fcout = F.relu(fcout)
            fcout = self.fc3(fcout)
            fcout = fcout.view(batch_size,2*self.output_channels_num, 2*self.input_channels_num)
            lstm_outputs.append(fcout)
        
        output = torch.stack(lstm_outputs, dim=1) 
        
        return output  
      
    def compute_stft(self, x):
        """Compute STFT for each channel separately and stack results."""
        batch_size, input_channels_num, time_steps = x.shape
        stft_list = []
        for c in range(input_channels_num):
            stft = torch.stft(x[:, c, :], n_fft=self.n_fft, win_length=self.n_fft, hop_length=self.hop_length, window=self.hann_window.to(x.device), return_complex=True)  # (batch, freq, time)
            stft_list.append(stft)
        return torch.stack(stft_list, dim=3)  # (batch, freq, time, channels)