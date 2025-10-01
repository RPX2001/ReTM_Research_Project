import torch
import torch.nn as nn
import torch.nn.functional as F

class FrequencyLSTMNet(nn.Module):
    def __init__(self, n_fft, hop_length, num_frequencies, num_channels, lstm_hidden_size, lstm_layers, num_output_channels):
        super(FrequencyLSTMNet, self).__init__()
        self.num_frequencies = num_frequencies
        self.num_channels = num_channels
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_layers = lstm_layers
        self.num_output_channels = num_output_channels
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        self.layer_norms = nn.ModuleList([nn.LayerNorm(4*lstm_hidden_size) for _ in range(num_frequencies)])
        self.lstm  = nn.LSTM(input_size = 2*num_channels, hidden_size = 2*lstm_hidden_size, num_layers=lstm_layers, batch_first=True, bidirectional=True)
        
        # Fully connected layers
        self.fc1 = nn.Linear(4*lstm_hidden_size, 8*(self.num_output_channels*(self.num_channels-self.num_output_channels)))
        self.fc2 = nn.Linear(8*(self.num_output_channels*(self.num_channels-self.num_output_channels)), 16*(self.num_output_channels*(self.num_channels-self.num_output_channels)))
        self.fc3 = nn.Linear(16*(self.num_output_channels*(self.num_channels-self.num_output_channels)), 4*(self.num_output_channels*(self.num_channels-self.num_output_channels)))        
    
    def forward(self, x):
        
        batch_size, num_channels, time_steps = x.shape
        stft = self.compute_stft(x) # batch, freq, time, channels, 2
        time_dim = stft.shape[2]  
        lstm_direct_outs = torch.zeros(batch_size, self.num_frequencies, 4 * self.lstm_hidden_size).to(x.device)  # (batch, freq, time, 2 * lstm_hidden_size)

        lstm_outputs = []
        for f in range(self.num_frequencies):
            lstm_input = torch.view_as_real(stft[:, f, :, :]).reshape(batch_size, time_dim, -1)  # (batch, STFT_time, (num_channels * 2))
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
            fcout = fcout.view(batch_size,2*self.num_output_channels, 2*(self.num_channels-self.num_output_channels))
            lstm_outputs.append(fcout)
        
        output = torch.stack(lstm_outputs, dim=1) 
        
        return output  
      
    def compute_stft(self, x):
        """Compute STFT for each channel separately and stack results."""
        batch_size, num_channels, time_steps = x.shape
        stft_list = []
        for c in range(num_channels):
            stft = torch.stft(x[:, c, :], n_fft=self.n_fft, win_length=self.n_fft, hop_length=self.hop_length, window=torch.hann_window(self.n_fft).to(x.device), return_complex=True)  # (batch, freq, time)
            stft_list.append(stft)
        return torch.stack(stft_list, dim=3)  # (batch, freq, time, channels, 2)