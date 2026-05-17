import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import os
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d  
import time
import torchaudio.transforms as T

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
      
      
def log_ratio_error(predicted, target):
    """
    Computes the mean of the log-ratio error as specified.
    
    Parameters:
    - predicted: Tensor of predicted magnitudes, shape (freq, time)
    - target: Tensor of target magnitudes, shape (freq, time)
    
    Returns:
    - Mean log-ratio error
    """
    
    predicted_img = predicted[3:, :,:]
    predicted_real = predicted[:3, :,:]
    target_img = target[3:, :,:]
    target_real = target[:3, :,:]
    predicted = predicted_real + 1j * predicted_img
    target = target_real + 1j * target_img
    # Compute the squared error
    squared_error = (abs(predicted - target)) ** 2
    
    # Normalize by the square of the target
    normalized_error = squared_error / (abs(target)** 2 + 1e-10)
    
    # Take log10 and then multiply by 10
    log_error = 10 * torch.log10(normalized_error + 1e-10)  # Adding a small value for numerical stability
    
    # Compute the mean across time (axis=1) and return the final mean error
    mean_log_ratio_error = torch.mean(log_error, dim=-1)
    
    
    return mean_log_ratio_error

def snr_loss(output: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute negative SNR loss for signals shaped [channels, signal_length].
    Loss = -10 * log10( signal_power / noise_power ), averaged over channels.
    """
    noise = target - output
    signal_power = torch.sum(target ** 2, dim=1)
    noise_power = torch.sum(noise ** 2, dim=1) + eps  # avoid division by zero
    snr = 10 * torch.log10(signal_power / noise_power)
    return -snr.mean()


def stft_error(predicted, target):
    """
    Computes the mean of the log-ratio error as specified.
    
    Parameters:
    - predicted: Tensor of predicted magnitudes, shape (freq, time)
    - target: Tensor of target magnitudes, shape (freq, time)
    
    Returns:
    - Mean log-ratio error
    """

    # Compute the squared error
    squared_error = (abs(predicted - target)) ** 2
    
    # Normalize by the square of the target
    normalized_error = squared_error / (abs(target)** 2 + 1e-10)
    
    # Take log10 and then multiply by 10
    log_error = 10 * torch.log10(normalized_error + 1e-10)  # Adding a small value for numerical stability
    
    # Compute the mean across time (axis=1) and return the final mean error
    mean_log_ratio_error = torch.mean(log_error)
    
    
    return mean_log_ratio_error



def get_training_example_paths(dataset_path):   
    """Returns a shuffled list of example paths without loading them"""
    example_paths = [os.path.join(dataset_path, folder) for folder in os.listdir(dataset_path)]
    return example_paths  # Do not shuffle here; shuffle per epoch in training loop

def load_example(example_path):
    """Loads a single example given its path"""
    recordings = []
    target_recordings = []
    
    # Load normal recordings (mic_4, mic_5, mic_6, mic_7)
    for i in range(4, 8):
        filepath = os.path.join(example_path, f"mic_{i}.wav")
        if os.path.isfile(filepath):
            waveform, _ = torchaudio.load(filepath)
            recordings.append(waveform)
    
    # Load target recordings (mic_1, mic_2, mic_3)
    for i in range(1, 4):
        filepath = os.path.join(example_path, f"mic_{i}.wav")
        if os.path.isfile(filepath):
            waveform, _ = torchaudio.load(filepath)
            target_recordings.append(waveform)
    
    if recordings and target_recordings:
        multichannel_audio = torch.stack(recordings, dim=0).squeeze(1)
        target_audio = torch.stack(target_recordings, dim=0).squeeze(1)
        min_length = min(multichannel_audio.shape[1], target_audio.shape[1])
        multichannel_audio, target_audio = multichannel_audio[:, :min_length], target_audio[:, :min_length]
        #multichannel_audio = multichannel_audio.unsqueeze(0)
        #target_audio = target_audio.unsqueeze(0)
        return multichannel_audio, target_audio
    return None  # Skip if loading failed

def snr_loss(output: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute negative SNR loss for signals shaped [channels, signal_length].
    Loss = -10 * log10( signal_power / noise_power ), averaged over channels.
    """
    noise = target - output
    signal_power = torch.sum(target ** 2, dim=1)
    noise_power = torch.sum(noise ** 2, dim=1) + eps  # avoid division by zero
    snr = 10 * torch.log10(signal_power / noise_power)
    return -snr.mean()

# Example usage
num_channels = 7
num_frequencies = 4097
lstm_hidden_size = num_channels
lstm_layers = 1
num_output_channels = 3
n_fft = 2*(num_frequencies-1)
win_length = n_fft

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = FrequencyLSTMNet(num_frequencies=num_frequencies, num_channels=num_channels, num_output_channels=num_output_channels, lstm_hidden_size=lstm_hidden_size, lstm_layers=lstm_layers, n_fft=2**13, hop_length = 2**13//2).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.01) 

dataset_path = "train_dataset_A_10"
n_epochs = 500
checkpoint = 1
train_example_paths = get_training_example_paths(dataset_path)
training_mode = 1
epoch = 0
time_weight = 1
stft_weight = 1


if checkpoint== 1 and training_mode==1:
  print("Loading checkpoint...")
  epoch = torch.load(f"checkpoint_{dataset_path}.pth")["epoch"]
  optimizer.load_state_dict(torch.load(f"checkpoint_{dataset_path}.pth")["optimizer_state_dict"])
  model.load_state_dict(torch.load(f"checkpoint_{dataset_path}.pth")["model_state_dict"])
  for param_group in optimizer.param_groups:
    param_group['lr'] = 0.0025
  print(f"Checkpoint loaded. Resuming from epoch {epoch+1} with learning rate {optimizer.param_groups[0]['lr']}")

recordings = []

for example_path in train_example_paths:
  multichannel_audio, target_audio = load_example(example_path)
  recordings_example = torch.cat((target_audio, multichannel_audio), dim=0)
  recordings.append(recordings_example)

recordings = torch.stack(recordings, dim = 0)
batch_size = recordings.shape[0]
recordings = recordings/abs(recordings).amax(axis=(1,2), keepdims=True)

if training_mode==1:
  while epoch < n_epochs:
 
      running_loss = 0
      model.train()
      
	      
      optimizer.zero_grad()
      
      input_data = recordings[:,:,:].to(device)
      retm = model(input_data)
      retm_to_save = retm.mean(axis=0).unsqueeze(0)
      retm = retm.mean(axis=0).unsqueeze(0).repeat(batch_size, 1, 1, 1)
      target_stft = model.compute_stft(input_data[:,0:3,:])  #(batch, freq, time, channels, 2)
      batch_size, freq, time_dim, _ = target_stft.shape
      target_stft = torch.view_as_real(target_stft).reshape(batch_size, freq, time_dim, -1)
      input_stft = model.compute_stft(input_data[:,3:,:])
      input_stft = torch.view_as_real(input_stft).reshape(batch_size, freq, time_dim, -1)
      output_stft = torch.einsum('bfnc,bftc->bftn', retm, input_stft)
      output_stft = output_stft[:,:,:,0::2] + 1j*output_stft[:,:,:,1::2]
      target_stft = target_stft[:,:,:,0::2] + 1j*target_stft[:,:,:,1::2]
      output_stft = output_stft.permute(0, 3, 1, 2)  # (batch, time, freq, channels)
      target_stft = target_stft.permute(0, 3, 1, 2)  # (batch, time, freq, channels)
      output_stft = output_stft.reshape(batch_size*3, freq, time_dim)  # (batch, freq * channels, time)
      target_stft = target_stft.reshape(batch_size*3, freq, time_dim)  # (batch, freq * channels, time)

      output_time = torch.istft(
                        output_stft, 
                        n_fft=n_fft,
                        hop_length=win_length//2,
                        win_length=win_length,
                        window = torch.hann_window(n_fft).to(device)
                    )
      target_time = torch.istft(
                        target_stft, 
                        n_fft=n_fft,
                        hop_length=win_length//2,
                        win_length=win_length,
                        window = torch.hann_window(n_fft).to(device)
                    )

      loss =  time_weight*snr_loss(output_time, target_time) + stft_weight*stft_error(output_stft, target_stft)

      loss.backward()  
      optimizer.step()
      
      running_loss += loss.item()
      
      print(f"Epoch {epoch+1}/{n_epochs} Loss: {running_loss}", snr_loss(output_time, target_time).item(), stft_error(output_stft, target_stft).item())
      epoch += 1
    
  checkpoint = {
          'epoch': n_epochs,
          'model_state_dict': model.state_dict(),
          'optimizer_state_dict': optimizer.state_dict()
    }
  torch.save(checkpoint, f"checkpoint_{dataset_path}.pth")
  torch.save(retm_to_save, f'retm_{dataset_path}.pt')


if training_mode==0 and checkpoint==1:
    model.load_state_dict(torch.load(f"checkpoint_{dataset_path}.pth")["model_state_dict"])
    if not os.path.exists(f'retm_{dataset_path}.pt'):
        raise FileNotFoundError(f"retm_{dataset_path}.pt file not found. Please ensure the file exists before running in test mode.")
    retm_to_save = torch.load(f'retm_{dataset_path}.pt')

model.eval()

test_example_paths = get_training_example_paths("train_dataset_A")
print(model)
i=0
for example_path in test_example_paths:
    multichannel_audio, target_audio = load_example(example_path)
    recordings_example = torch.cat((target_audio, multichannel_audio), dim=0).unsqueeze(0)

    input_data = recordings_example[0:1,3:,:].to(device)

    output_stft = model(input_data)
    result = output_stft.squeeze(0)
    # retm = torch.view_as_complex(retm.reshape(1, num_frequencies, num_other_channels, 2).contiguous())
    target_stft = model.compute_stft(input_data[:,:3,:])  #(batch, freq, time, channels, 2)
    target_stft = target_stft.squeeze(0)
    freq, time_dim, _ = target_stft.shape
    target_stft = torch.view_as_real(target_stft).reshape(freq, time_dim, -1)
    error = log_ratio_error(result.permute(2,0,1), target_stft.permute(2,0,1))
    if i==0:
        target_test_output = target_stft.permute(2,0,1)
        test_output =result.permute(2,0,1)
    else:
        target_test_output = torch.cat((target_test_output, target_stft.permute(2,0,1)), dim=2)
        test_output = torch.cat((test_output, result.permute(2,0,1)), dim=2)
    i+=1



# Test example

folder_path = ""  # Specify test folder path here
filepaths = ["mic_1.wav", "mic_2.wav","mic_3.wav", "mic_4.wav", "mic_5.wav", "mic_6.wav", "mic_7.wav"]

recordings = []
for filepath in filepaths:
    full_path = os.path.join(folder_path, filepath) if folder_path else filepath
    waveform, sr = torchaudio.load(full_path)
    recordings.append(waveform[:])
batch_size = 1

recordings_array = np.vstack(recordings[:])
recordings_array = recordings_array.reshape(1,recordings_array.shape[0],recordings_array.shape[1])
recordings_array = torch.from_numpy(recordings_array)
recordings_array = recordings_array.repeat(batch_size, 1, 1)
recordings_array_1 = recordings_array.to(device)

input_data = recordings_array_1[:,:,:].to(device)

target_stft = model.compute_stft(input_data[:,0:3,:])  #(batch, freq, time, channels, 2)
batch_size, freq, time_dim, _ = target_stft.shape
target_stft = torch.view_as_real(target_stft).reshape(batch_size, freq, time_dim, -1)

input_stft = model.compute_stft(input_data[:,3:,:])
input_stft = torch.view_as_real(input_stft).reshape(1, freq, time_dim, -1)

output_stft = torch.einsum('bfnc,bftc->bftn', retm_to_save, input_stft)

result = output_stft.squeeze(0)
target_stft = target_stft.squeeze(0)

result_to_inverse = result[:,:,0::2] + 1j*result[:,:,1::2]
target_to_inverse = target_stft[:,:,0::2] + 1j*target_stft[:,:,1::2]
output_time = torch.istft(
                        result_to_inverse.permute(2, 0, 1), 
                        n_fft=n_fft,
                        hop_length=win_length//2,
                        win_length=win_length,
                        window=torch.hann_window(n_fft).to(device)
                    )
target_time = torch.istft(
                target_to_inverse.permute(2, 0, 1), 
                n_fft=n_fft,
                hop_length=win_length//2,
                win_length=win_length,
                window=torch.hann_window(n_fft).to(device)
            )

np.save("target_LSTM_A.npy", target_time.detach().cpu().numpy(), sr)
np.save("output_LSTM_A.npy", output_time.detach().cpu().numpy(), sr)  


error = log_ratio_error(result.permute(2,0,1), target_stft.permute(2,0,1))
error = error.cpu().detach().numpy()

		
