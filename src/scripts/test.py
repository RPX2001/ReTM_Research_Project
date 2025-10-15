import torch
import torch.nn as nn
import torch.nn.functional as F
import unittest
from types import SimpleNamespace

# [Insert the FrequencyLSTMNet class definition here, including all methods]
# ... (The provided class definition goes here) ...

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

        self.num_frequencies = self.n_fft//2 + 1    # In the original code, this was 4097, which is 2**13/2 + 1 where n_fft = 2**13
        self.lstm_hidden_size = self.input_channels_num
        self.hop_length = self.n_fft//2             # 2**13//2
        self.hann_window = torch.hann_window(self.n_fft)
        
        # Note: The original code uses 4*self.lstm_hidden_size for LayerNorm, 
        # but the LSTM output size is 4*self.lstm_hidden_size (2*hidden_size for bidirectional).
        self.layer_norms = nn.ModuleList([nn.LayerNorm(4*self.lstm_hidden_size) for _ in range(self.num_frequencies)])
        self.lstm  = nn.LSTM(input_size = 2*self.input_channels_num, hidden_size = 2*self.lstm_hidden_size, num_layers=self.lstm_layers, batch_first=True, bidirectional=True)
        
        # Fully connected layers
        in_fc_size = 4*self.lstm_hidden_size
        out_fc_size = 4 * (self.output_channels_num * (self.input_channels_num - self.output_channels_num))
        
        self.fc1 = nn.Linear(in_fc_size, 8*(self.output_channels_num*(self.input_channels_num-self.output_channels_num)))
        self.fc2 = nn.Linear(8*(self.output_channels_num*(self.input_channels_num-self.output_channels_num)), 16*(self.output_channels_num*(self.input_channels_num-self.output_channels_num)))
        self.fc3 = nn.Linear(16*(self.output_channels_num*(self.input_channels_num-self.output_channels_num)), out_fc_size)        
    
    def forward(self, x):
        
        batch_size, input_channels_num, time_steps = x.shape
        stft = self.compute_stft(x) 
        time_dim = stft.shape[2]    
        
        # This tensor is declared but not actually used later in the loop to store outputs, 
        # so it's commented out/ignored for flow testing.
        # lstm_direct_outs = torch.zeros(batch_size, self.num_frequencies, 4 * self.lstm_hidden_size).to(x.device)  

        lstm_outputs = []
        for f in range(self.num_frequencies):
            # (batch, STFT_time, channels) -> view_as_real (batch, STFT_time, channels, 2) -> reshape (batch, STFT_time, channels*2)
            lstm_input = torch.view_as_real(stft[:, f, :, :]).reshape(batch_size, time_dim, -1)  
            
            # lstm_out shape: (batch, STFT_time, 2*bidirectional*hidden_size) = (batch, STFT_time, 4*lstm_hidden_size)
            lstm_out, _ = self.lstm(lstm_input) 
            lstm_out = self.layer_norms[f](lstm_out)
            
            # Average across the time dimension: (batch, 4*lstm_hidden_size)
            lstm_out = lstm_out.mean(axis=1) 
            
            # This line corresponds to the unused lstm_direct_outs
            # lstm_direct_outs[:, f, :] = lstm_out  

            # fc_input shape: (batch, 4*lstm_hidden_size)
            # fc1/fc2/fc3 operations
            fcout = self.fc1(lstm_out)
            fcout = F.relu(fcout)
            fcout = self.fc2(fcout)
            fcout = F.relu(fcout)
            fcout = self.fc3(fcout)
            
            # fcout shape: (batch, 2*output_channels_num, 2*(input_channels_num-output_channels_num))
            fcout = fcout.view(batch_size, 2*self.output_channels_num, 2*(self.input_channels_num-self.output_channels_num))
            lstm_outputs.append(fcout)
        
        # output shape: (batch, num_frequencies, 2*output_channels_num, 2*(input_channels_num-output_channels_num))
        output = torch.stack(lstm_outputs, dim=1) 
        
        return output  
      
    def compute_stft(self, x):
        """Compute STFT for each channel separately and stack results."""
        batch_size, input_channels_num, time_steps = x.shape
        stft_list = []
        for c in range(input_channels_num):
            stft = torch.stft(x[:, c, :], n_fft=self.n_fft, win_length=self.n_fft, hop_length=self.hop_length, window=self.hann_window.to(x.device), return_complex=True)  # (batch, freq, time)
            stft_list.append(stft)
        # return shape: (batch, freq, time, channels) complex tensor
        return torch.stack(stft_list, dim=3)


# --- TEST SCRIPT ---
# Define the required parameters using SimpleNamespace to mock the 'args' object
test_args = SimpleNamespace(
    n_fft=1024,                      # Standard FFT size
    lstm_layers=2,                   # Number of LSTM layers
    input_mics_numbers=[1, 2, 3, 4], # 4 input channels
    output_mics_numbers=[5, 6]       # 2 output channels
)

# Derived parameters for verification
BATCH_SIZE = 4
INPUT_CHANNELS = len(test_args.input_mics_numbers)
OUTPUT_CHANNELS = len(test_args.output_mics_numbers)
NUM_SAMPLES = 16000 # A reasonable number of audio samples
NUM_FREQUENCIES = test_args.n_fft // 2 + 1 # 1024/2 + 1 = 513
LSTM_HIDDEN_SIZE = INPUT_CHANNELS # which is 4
TIME_DIM = (NUM_SAMPLES - test_args.n_fft) // (test_args.n_fft // 2) + 1 # STFT time steps

# Expected final output shape
# The shape before stacking by frequency is: 
# (batch, 2*output_channels_num, 2*(input_channels_num-output_channels_num))
# With the stack (dim=1) for frequencies:
# (batch, num_frequencies, 2*output_channels_num, 2*(input_channels_num-output_channels_num))
EXPECTED_OUT_SHAPE = (
    BATCH_SIZE,
    NUM_FREQUENCIES,
    2 * OUTPUT_CHANNELS,         # 2 * 2 = 4
    2 * (INPUT_CHANNELS - OUTPUT_CHANNELS) # 2 * (4 - 2) = 4
)


def run_test():
    # 1. Instantiate the model
    print(f"Instantiating model with: n_fft={test_args.n_fft}, input_ch={INPUT_CHANNELS}, output_ch={OUTPUT_CHANNELS}")
    try:
        model = FrequencyLSTMNet(test_args)
    except AttributeError as e:
        print(f"Error during model instantiation: {e}")
        return

    # 2. Create a dummy input tensor: (batch_size, input_channels, samples)
    # Using float32 for standard audio processing
    dummy_input = torch.randn(BATCH_SIZE, INPUT_CHANNELS, NUM_SAMPLES, dtype=torch.float32)
    print(f"Dummy input shape: {list(dummy_input.shape)}")

    # 3. Perform the forward pass
    print("Running forward pass...")
    try:
        output = model(dummy_input)
    except Exception as e:
        print(f"An error occurred during the forward pass: {e}")
        return

    # 4. Check the output
    actual_shape = tuple(output.shape)
    print(f"\nExpected output shape: {list(EXPECTED_OUT_SHAPE)}")
    print(f"Actual output shape: {list(actual_shape)}")

    assert actual_shape == EXPECTED_OUT_SHAPE, "Output shape mismatch!"
    print("\n✅ Test Passed: Output shape is correct.")

if __name__ == '__main__':
    run_test()