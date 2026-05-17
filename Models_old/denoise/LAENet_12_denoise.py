import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import os
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d  # For smoothing
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
        # self.transform = T.Spectrogram(n_fft=self.n_fft, win_length=self.n_fft, hop_length=self.n_fft//2, window_fn=lambda n_fft: torch.hamming_window(self.n_fft, periodic=True), power=None) 
        
    def forward(self, x):
        # x shape: (batch, channels, time)
        batch_size, num_channels, time_steps = x.shape
        # print(x.shape)
        # Compute STFT
        stft = self.compute_stft(x) # batch, freq, time, channels, 2
        time_dim = stft.shape[2]  # Correct time dimension after STFT
        lstm_direct_outs = torch.zeros(batch_size, self.num_frequencies, 4 * self.lstm_hidden_size).to(x.device)  # (batch, freq, time, 2 * lstm_hidden_size)

        lstm_outputs = []
        for f in range(self.num_frequencies):
            lstm_input = torch.view_as_real(stft[:, f, :, :]).reshape(batch_size, time_dim, -1)  # (batch, STFT_time, (num_channels * 2))
            lstm_out, _ = self.lstm(lstm_input)
            # lstm_out = lstm_out*mean**2
            lstm_out = self.layer_norms[f](lstm_out)
            
            lstm_out = lstm_out.mean(axis=1)
            lstm_direct_outs[:, f, :] = lstm_out  # (batch, freq, 2 * lstm_hidden_size)
            lstm_out = lstm_out.view(batch_size, -1)  # (batch, 2 * lstm_hidden_size)
            # print(f,lstm_out[0,:])
            # print(lstm_out.shape)
            fcout = self.fc1(lstm_out)
            fcout = F.relu(fcout)
            fcout = self.fc2(fcout)
            fcout = F.relu(fcout)
            fcout = self.fc3(fcout)
            fcout = fcout.view(batch_size,2*self.num_output_channels, 2*(self.num_channels-self.num_output_channels))
            lstm_outputs.append(fcout)
        
        
        
        
        #print(lstm_outputs[50])
        # Stack outputs from all frequencies
        output = torch.stack(lstm_outputs, dim=1)  # (batch, freq, lstm_hidden_size)
        
        return output  # (batch, freq, 2 * num_other_channels)
      
    def compute_stft(self, x):
        """Compute STFT for each channel separately and stack results."""
        batch_size, num_channels, time_steps = x.shape
        # print(x.shape)
        stft_list = []
        for c in range(num_channels):
            stft = torch.stft(x[:, c, :], n_fft=self.n_fft, win_length=self.n_fft, hop_length=self.hop_length, window=torch.hann_window(self.n_fft).to(x.device), return_complex=True)  # (batch, freq, time)
            # stft = torch.view_as_real(stft)  # (batch, freq, time, 2)
            # stft = self.transform(x[:, c, :])
            # print(stft.shape)
            stft_list.append(stft)
        # return stft
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
    # predicted_img = predicted[5:, :,:]
    # predicted_real = predicted[:5, :,:]
    # target_img = target[5:, :,:]
    # target_real = target[:5, :,:]
    # predicted = predicted_real + 1j * predicted_img
    # target = target_real + 1j * target_img

    # Compute the squared error
    squared_error = (abs(predicted - target)) ** 2
    
    # Normalize by the square of the target
    normalized_error = squared_error / (abs(target)** 2 + 1e-10)
    
    # Take log10 and then multiply by 10
    log_error = 10 * torch.log10(normalized_error + 1e-10)  # Adding a small value for numerical stability
    print("shape is ", log_error.shape)
    # Compute the mean across time (axis=1) and return the final mean error
    mean_log_ratio_error = torch.mean(log_error, dim=-1)
    
    
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
    for i in range(6, 13):
        filepath = os.path.join(example_path, f"mic_{i}.wav")
        if os.path.isfile(filepath):
            waveform, _ = torchaudio.load(filepath)
            recordings.append(waveform)
    
    # Load target recordings (mic_1, mic_2, mic_3)
    for i in range(1, 6):
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



# Example usage
num_channels = 12
num_frequencies = 4097
lstm_hidden_size = num_channels
lstm_layers = 1
num_output_channels = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# audio_input = torch.randn(batch_size, num_channels, 2**13*2).to(device)  # Simulated 1-second audio with 16kHz
model = FrequencyLSTMNet(num_frequencies=num_frequencies, num_channels=num_channels, num_output_channels=num_output_channels, lstm_hidden_size=lstm_hidden_size, lstm_layers=lstm_layers, n_fft=2**13, hop_length = 2**13//2).to(device)


filepaths = ["mic_1.wav", "mic_2.wav","mic_3.wav", "mic_4.wav", "mic_5.wav", "mic_6.wav", "mic_7.wav", "mic_8.wav", "mic_9.wav", "mic_10.wav", "mic_11.wav", "mic_12.wav"]

recordings = []
for filepath in filepaths:
    waveform, _ = torchaudio.load(filepath)
    recordings.append(waveform[:])
    # print(filepath, waveform[:])
batch_size = 1
# print(recordings)

recordings_array = np.vstack(recordings[:])
# print(recordings_array)
recordings_array = recordings_array.reshape(1,recordings_array.shape[0],recordings_array.shape[1])
recordings_array = torch.from_numpy(recordings_array)
recordings_array = recordings_array.repeat(batch_size, 1, 1)
# recordings_array = recordings_array[:,:,:2**13]
recordings_array_1 = recordings_array.to(device)
criterion = nn.MSELoss()  # Mean Squared Error Loss for regression
# criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.001) 
alpha = 100

dataset_path = "train_dataset_C_10_denoise_AWGN"
n_epochs = 4400
checkpoint = 1
train_example_paths = get_training_example_paths(dataset_path)
training_mode = 0
epoch = 0
decay_epoch = 101
decay_factor = 0.1


if checkpoint== 1 and training_mode==1:
  epoch = torch.load("final_checkpoint_C.pth")["epoch"]
  optimizer.load_state_dict(torch.load("final_checkpoint_C.pth")["optimizer_state_dict"])
  model.load_state_dict(torch.load("final_checkpoint_C.pth")["model_state_dict"])

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
      if epoch + 1 == decay_epoch:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= decay_factor
        print(f"Learning rate decreased to {optimizer.param_groups[0]['lr']}")
 
      running_loss = 0
   # for i in range(recordings_array.shape[0]):
      model.train()
      
	      
      optimizer.zero_grad()
      #recordings_array = torch.cat((target_audio, multichannel_audio), dim=1)
      # input_data = recordings_array.to(device)
      
      input_data = recordings[:,:,:].to(device)
      # print(input_data.shape)
      retm = model(input_data)
      retm_to_save = retm.mean(axis=0).unsqueeze(0)
      retm = retm.mean(axis=0).unsqueeze(0).repeat(batch_size, 1, 1, 1)
    #   print(output_stft.shape)
      target_stft = model.compute_stft(input_data[:,0:5,:])  #(batch, freq, time, channels, 2)
      batch_size, freq, time_dim, _ = target_stft.shape
      target_stft = torch.view_as_real(target_stft).reshape(batch_size, freq, time_dim, -1)
      input_stft = model.compute_stft(input_data[:,5:,:])
      input_stft = torch.view_as_real(input_stft).reshape(batch_size, freq, time_dim, -1)
    #   print(retm.shape)
      output_stft = torch.einsum('bfnc,bftc->bftn', retm, input_stft)
    #   print(target_stft.shape, output_stft.shape)
      # batch_size = output_stft.shape[0]
      # output_norm = torch.norm(output_stft, dim=3, keepdim=True)
      
      # batch_size = target_stft.shape[0]
      # target_norm = torch.norm(target_stft, dim=3, keepdim=True)
      loss = criterion(output_stft[:,:,:,:], target_stft[:,:,:,:])
      
      # error = log_ratio_error_img(result[0,:].squeeze(), target_stft[0,:].squeeze())
      # print(error)
    #   print("losses",criterion(result_stft, target_stft), criterion(target_time, result_time) )
      # print("loss is", loss.item())
      loss.backward()  
      optimizer.step()
      
      running_loss += loss.item()
      
      print(f"Epoch {epoch+1}/{n_epochs} Loss: {running_loss}")
      epoch += 1
    
  checkpoint = {
          'epoch': n_epochs,
          'model_state_dict': model.state_dict(),
          'optimizer_state_dict': optimizer.state_dict()
    }
  torch.save(checkpoint, "final_checkpoint_C.pth")
  torch.save(retm_to_save, 'retm_C_AWGN.pt')


if training_mode==0 and checkpoint==1:
    model.load_state_dict(torch.load("final_checkpoint_C.pth")["model_state_dict"])
    retm_to_save = torch.load('retm_C_AWGN.pt')

model.eval()
    
    
denoising_file = "rec_scen_C.wav"

# Load the multi-channel file
waveform, sample_rate = torchaudio.load(denoising_file)  # waveform shape: (channels, time)

# For example, extract channels 0 and 2
output_indices = [0, 1, 2, 3, 4]
input_indices = [5, 6, 7, 8, 9, 10, 11]
target_recordings = waveform[output_indices, :]
multichannel_audio = waveform[input_indices, :]
multichannel_audio = multichannel_audio.unsqueeze(0)
target_recordings = target_recordings.unsqueeze(0)

input_data = multichannel_audio.to(device)
target_stft = model.compute_stft(target_recordings.to(device))  #(batch, freq, time, channels, 2)
batch_size, freq, time_dim, _ = target_stft.shape
# target_stft = torch.view_as_real(target_stft).reshape(batch_size, freq, time_dim, -1)

input_stft = model.compute_stft(input_data)
input_stft = torch.view_as_real(input_stft).reshape(1, freq, time_dim, -1)

output_stft = torch.einsum('bfnc,bftc->bftn', retm_to_save, input_stft)
output_stft = output_stft[:,:,:,0::2] + 1j*output_stft[:,:,:,1::2]
result = output_stft.squeeze(0)
target_stft = target_stft.squeeze(0)
print(result.shape, target_stft.shape)
n_fft = 2**13
win_length = n_fft

denoised_stft = target_stft - result
denoised_stft = denoised_stft.permute(2, 0, 1)  # (time, batch, freq, channels)
denoised_sig = torch.istft(
    denoised_stft, 
    n_fft=n_fft,
    hop_length=win_length//2,
    win_length=win_length
)

output_sig = torch.istft(
    target_stft.permute(2, 0, 1), 
    n_fft=n_fft,
    hop_length=win_length//2,
    win_length=win_length
)

target_sig = torch.istft(
    result.permute(2, 0, 1), 
    n_fft=n_fft,
    hop_length=win_length//2,
    win_length=win_length
)

# Save output and target as numpy arrays
output_numpy = output_sig.detach().cpu().numpy()
target_numpy = target_sig.detach().cpu().numpy()

# Save to .npy files
np.save("output_scenario_C.npy", output_numpy)
np.save("target_scenario_C.npy", target_numpy)