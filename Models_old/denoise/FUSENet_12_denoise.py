import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d  # For smoothing
import random
import torch.nn.functional as F

# Define the model class
class MultiChannelConvolutionModel(nn.Module):
    def __init__(self, filter_length):
        super(MultiChannelConvolutionModel, self).__init__()
        
        # Define 4 convolutional filters for the 4 channels of audio
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=5, kernel_size=filter_length, bias=False)
        self.conv2 = nn.Conv1d(in_channels=1, out_channels=5, kernel_size=filter_length, bias=False)
        self.conv3 = nn.Conv1d(in_channels=1, out_channels=5, kernel_size=filter_length, bias=False)
        self.conv4 = nn.Conv1d(in_channels=1, out_channels=5, kernel_size=filter_length, bias=False)
        self.conv5 = nn.Conv1d(in_channels=1, out_channels=5, kernel_size=filter_length, bias=False)
        self.conv6 = nn.Conv1d(in_channels=1, out_channels=5, kernel_size=filter_length, bias=False)
        self.conv7 = nn.Conv1d(in_channels=1, out_channels=5, kernel_size=filter_length, bias=False)
        
    def forward(self, x):
        # Apply the convolution filters on each channel (dim=1)
        conv1_out = self.conv1(x[:, 0:1, :])  # Only take the 1st channel for convolution
        conv2_out = self.conv2(x[:, 1:2, :])  # Only take the 2nd channel for convolution
        conv3_out = self.conv3(x[:, 2:3, :])  # Only take the 3rd channel for convolution
        conv4_out = self.conv4(x[:, 3:4, :])  # Only take the 4th channel for convolution
        conv5_out = self.conv5(x[:, 4:5, :])  # Only take the 1st channel for convolution
        conv6_out = self.conv6(x[:, 5:6, :])  # Only take the 4th channel for convolution
        conv7_out = self.conv7(x[:, 6:7, :])  # Only take the 1st channel for convolution
        
        # Average the outputs of all channels
        out = (conv1_out + conv2_out + conv3_out + conv4_out + conv5_out + conv6_out + conv7_out)
        
        # Return the averaged result
        return out # Remove channel dimension

def log_error(y_c, y_c_star, n_fft, hop_length,win_length):
  """
  Computes the loss based on MSE between STFT magnitudes.

  Args:
  y_c (Tensor): The original signal (1D time-domain)
  y_c_star (Tensor): The estimated signal (1D time-domain)

  Returns:
  loss (Tensor): STFT-based MSE loss value
  """

  # # Ensure input is 1D
  # if y_c.dim() != 1 or y_c_star.dim() != 1:
  #     raise ValueError("Inputs must be 1D tensors (time-domain signals).")

  # Normalize signals using L2 norm
  # y_c_norm = y_c / (torch.norm(y_c, p=2) + 1e-8)
  # y_c_star_norm = y_c_star / (torch.norm(y_c_star, p=2) + 1e-8)
  
  y_c_norm = y_c
  y_c_star_norm = y_c_star

  # Compute STFT magnitudes
  Y_c = torch.stft(y_c_norm, n_fft=n_fft, hop_length=hop_length, 
                    win_length=win_length, return_complex=True)
  Y_c_star = torch.stft(y_c_star_norm, n_fft=n_fft, hop_length=hop_length, 
                        win_length=win_length, return_complex=True)
  
  # Convert STFT magnitudes to Mel spectrograms
  # Mel_Y_c = self.mel_filterbank(Y_c)
  # Mel_Y_c_star = self.mel_filterbank(Y_c_star)
  squared_error = (abs(Y_c_star - Y_c)) ** 2
    
  # Normalize by the square of the target
  normalized_error = squared_error / (abs(Y_c_star)** 2 + 1e-10)
  # Compute Mean Squared Error between magnitudes
  log_error = 10 * torch.log10(normalized_error + 1e-10)
  
  # mean_log_error = torch.mean(log_error, dim=-1)

  return log_error


import os
import random
import torch
import torchaudio

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
        multichannel_audio = multichannel_audio.unsqueeze(0)
        target_audio = target_audio.unsqueeze(0)
        return multichannel_audio, target_audio
    return None  # Skip if loading failed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create a model instance
context = 2**13//2
filter_length = int(2*context)+1  # Length of the convolution filters (can be adjusted)
model = MultiChannelConvolutionModel(filter_length).to(device)

# Define the loss function and optimizer
criterion = nn.MSELoss()  # Mean Squared Error Loss for regression
optimizer = optim.Adam(model.parameters(), lr=0.00001)


def get_overlapping_windows(recordings_array, window_size, stride):
    return recordings_array.unfold(dimension=2, size=window_size, step=stride).squeeze().permute(1, 0, 2)


fs = 16 # Sampling frequency, replace with your value
n_fft = 2**13

# Compute positive frequencies
frequencies = np.fft.fftfreq(n_fft, d=1/fs)
positive_frequencies = frequencies[frequencies >= 0]
positive_frequencies = np.append(positive_frequencies, abs(frequencies[n_fft//2]))

# print(len(positive_frequencies))
# print(positive_frequencies)

dataset_path = "train_dataset_C_10_denoise_AWGN"
valid_dataset_path = "valid_dataset_B_10"
mse_loss_history = []
validation_loss_history = []

training_mode = 0# 1 for training, 0 for testing
checkpoint = 1
valid = 0

train_example_paths = get_training_example_paths(dataset_path)
if valid==1:
    valid_example_paths = get_training_example_paths(valid_dataset_path)
    
n_epochs = 200
epoch = 0
if checkpoint== 1 and training_mode==1:
  epoch = torch.load("final_checpoint_denoise_C_AWGN.pth")["epoch"]
  optimizer.load_state_dict(torch.load("final_checpoint_denoise_C_AWGN.pth")["optimizer_state_dict"])
  model.load_state_dict(torch.load("final_checpoint_denoise_C_AWGN.pth")["model_state_dict"])
  
if training_mode==1:
    while epoch<n_epochs:
          epoch_loss = 0
          num_files = 0
          random.shuffle(train_example_paths)
          model.train(True)
          for example_path in train_example_paths:

            multichannel_audio, target_audio = load_example(example_path)
            pad_size = context
            multichannel_audio = F.pad(multichannel_audio, (pad_size, pad_size), mode='constant', value=0)
            target_audio = F.pad(target_audio, (pad_size, pad_size), mode='constant', value=0)
            
            multichannel_audio = get_overlapping_windows(multichannel_audio, 2**14, 2**13)
            target_audio = get_overlapping_windows(target_audio, 2**14, 2**13)
            running_loss = 0
            
            for i in range(1, multichannel_audio.shape[0]- 1):
              
              
              optimizer.zero_grad()
              input_data = multichannel_audio[i,:,:].unsqueeze(0)
              target = target_audio[i,:,context:-context].unsqueeze(0).to(device)
              output = model(input_data.to(device))
              # output = output.squeeze()
              
              # print(target)
              # print(output)

              loss = criterion(output, target) 
              
              loss.backward()
              optimizer.step()
              
              running_loss += loss.item()
            epoch_loss+= running_loss
            num_files +=1
          mse_loss_history.append(epoch_loss / (num_files*multichannel_audio.shape[0]))
              
          if valid==1:
          
            model.eval()
            validation_loss = 0
            with torch.no_grad():
                    num_files = 0
                    for example_path in valid_example_paths:  # Use a separate validation dataset here
                        multichannel_audio, target_audio = load_example(example_path)
                        pad_size = context
                        multichannel_audio = F.pad(multichannel_audio, (pad_size, pad_size), mode='constant', value=0)
                        target_audio = F.pad(target_audio, (pad_size, pad_size), mode='constant', value=0)
                        
                        multichannel_audio = get_overlapping_windows(multichannel_audio, 2**14, 2**13)
                        target_audio = get_overlapping_windows(target_audio, 2**14, 2**13)
                        
                        input_data = multichannel_audio[:,:,:]
                        target = target_audio[:,:,context:-context]
                        output = model(input_data)
                        

                        val_loss = criterion(output, target)
                        validation_loss += val_loss.item()
                        num_files += 1

                    validation_loss /= (num_files*multichannel_audio.shape[0])
                    validation_loss_history.append(validation_loss)

          print(f"Epoch {epoch+1}/{n_epochs} Train Loss: {epoch_loss / (num_files*multichannel_audio.shape[0])}")
          if valid==1:
              print(f"Validation Loss: {validation_loss}")
          epoch += 1


    # Save final checkpoint after all epochs
    checkpoint = {
        'epoch': n_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss_history': mse_loss_history,
        'val_loss_history': validation_loss_history
    }
    torch.save(checkpoint, "final_checpoint_denoise_C_AWGN.pth")

if training_mode==0 and checkpoint==1:
    model.load_state_dict(torch.load("final_checpoint_denoise_C_AWGN.pth")["model_state_dict"])
model.eval()
# input_data = recordings_array[:,3:,:]
# target = recordings_array[:,:3,:]
# output = model(input_data)

test_dataset_path = "test_dataset_C_10_denoise_AWGN"
test_loss = 0
j=0
test_example_paths = get_training_example_paths(test_dataset_path)
print(test_example_paths)
random.shuffle(test_example_paths)
print(test_example_paths)
for example_path in test_example_paths:
    # max_value = torch.max(torch.abs(torch.cat((multichannel_audio, target_audio), dim=0)))
    multichannel_audio, target_audio = load_example(example_path)
    pad_size = context
    
    multichannel_audio = F.pad(multichannel_audio, (pad_size, pad_size), mode='constant', value=0)   
    target_audio = F.pad(target_audio, (pad_size, pad_size), mode='constant', value=0)
    # Normalize the audio signals
    multichannel_audio = get_overlapping_windows(multichannel_audio, 2**14, 2**13)
    target_audio = get_overlapping_windows(target_audio, 2**14, 2**13)
    running_loss = 0
    
    input_data = multichannel_audio[:,:,:]
    target = target_audio[:,:,context:-context].to(device)
    
    output = model(input_data.to(device))
      # output = output.squeeze()
      
      
      # print(target)
      # print(output)

    loss = criterion(output, target)
    print("output shape is" ,output.shape)
    output = output.permute(1, 0, 2).reshape(5, -1)
    target = target.permute(1, 0, 2).reshape(5, -1)
    
    error = log_error(output, target, n_fft=2**13, win_length=2**13, hop_length=2**13//2)
      
    if j==0:
      previous_error = error
    
    else:
      previous_error = torch.cat((error, previous_error), dim=-1)

    
    test_loss += loss.item()
    j+=1

test_loss /= len(os.listdir(test_dataset_path)*multichannel_audio.shape[0])


output_numpy = output.detach().cpu().numpy()
target_numpy = target.detach().cpu().numpy()

# Save to .npy files
np.save("output_scenario_C.npy", output_numpy)
np.save("target_scenario_C.npy", target_numpy)

print(f"Test Loss: {test_loss:.4f}") 

denoising_file = "rec_scen_C.wav"

# Load the multi-channel file
waveform, sample_rate = torchaudio.load("rec_scen_C.wav")  # waveform shape: (channels, time)

# For example, extract channels 0 and 2
output_indices = [0, 1, 2, 3, 4]
input_indices = [5, 6, 7, 8, 9, 10, 11]
target_recordings = waveform[output_indices, :]# Add batch dimension
multichannel_audio = waveform[input_indices, :].unsqueeze(0)  # Add batch dimension
np.save('multichannel_audio_time_C.npy', multichannel_audio.cpu().detach())
pad_size = context
multichannel_audio = F.pad(multichannel_audio, (pad_size, pad_size), mode='constant', value=0)

multichannel_audio = get_overlapping_windows(multichannel_audio, 2**14, 2**13)

input_data = multichannel_audio[:,:,:]
output_sig = model(input_data.to(device))
output_sig = output_sig.permute(1, 0, 2).reshape(5, -1).cpu()
target_recordings= target_recordings[:,:len(output_sig[0])]

print(target_recordings.shape, output_sig.shape)
denoised_sig = target_recordings- output_sig

np.save('denoised_sig_time_C.npy', denoised_sig.cpu().detach())

torchaudio.save("denoised_sig_scenario_C_CNN_Time.wav", denoised_sig.detach(), sample_rate)
torchaudio.save("noisy_sig_scenario_C_CNN_Time.wav", target_recordings.detach(), sample_rate)