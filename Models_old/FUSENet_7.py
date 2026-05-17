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
        return out # Remove channel dimension

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

def log_error(y_c, y_c_star, n_fft, hop_length,win_length):
    """
    Computes the loss based on MSE between STFT magnitudes.

    Args:
    y_c (Tensor): The original signal (1D time-domain)
    y_c_star (Tensor): The estimated signal (1D time-domain)

    Returns:
    loss (Tensor): STFT-based MSE loss value
    """

    y_c_norm = y_c
    y_c_star_norm = y_c_star

    # Compute STFT magnitudes
    Y_c = torch.stft(y_c_norm, n_fft=n_fft, hop_length=hop_length, 
                    win_length=win_length, window=torch.hann_window(n_fft).to(y_c_norm.device), return_complex=True)
    Y_c_star = torch.stft(y_c_star_norm, n_fft=n_fft, hop_length=hop_length, 
                        win_length=win_length, window=torch.hann_window(n_fft).to(y_c_star_norm.device), return_complex=True)
    squared_error = (abs(Y_c_star - Y_c)) ** 2

    # Normalize by the square of the target
    normalized_error = squared_error / (abs(Y_c_star)** 2 + 1e-10)
    # Compute Mean Squared Error between magnitudes
    log_error = 10 * torch.log10(normalized_error + 1e-10)

    return log_error


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
        multichannel_audio = multichannel_audio.unsqueeze(0)
        target_audio = target_audio.unsqueeze(0)
        return multichannel_audio, target_audio
    return None  # Skip if loading failed

def get_overlapping_windows(recordings_array, window_size, stride):
    return recordings_array.unfold(dimension=2, size=window_size, step=stride).squeeze().permute(1, 0, 2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create a model instance
context = 2**13//2
filter_length = int(2*context)+1  # Length of the convolution filters 
model = MultiChannelConvolutionModel(filter_length).to(device)

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.0001) 
time_weight = 10
stft_weight = 1

fs = 16 # Sampling frequency
n_fft = 2**13

# Compute positive frequencies
frequencies = np.fft.fftfreq(n_fft, d=1/fs)
positive_frequencies = frequencies[frequencies >= 0]
positive_frequencies = np.append(positive_frequencies, abs(frequencies[n_fft//2]))

dataset_path = "train_dataset_A_AWGN"
valid_dataset_path = "valid_dataset_B_10"
test_dataset_path = "test_dataset_A_AWGN"

training_mode = False # 1 for training, 0 for testing
checkpoint = True
valid = False # whether to do validation

train_example_paths = get_training_example_paths(dataset_path)
if valid==1:
    valid_example_paths = get_training_example_paths(valid_dataset_path)
    
n_epochs = 50 
epoch = 0
if checkpoint and training_mode:
  epoch = torch.load(f"checkpoint_{dataset_path}.pth")["epoch"]
  optimizer.load_state_dict(torch.load(f"checkpoint_{dataset_path}.pth")["optimizer_state_dict"])
  model.load_state_dict(torch.load(f"checkpoint_{dataset_path}.pth")["model_state_dict"])


if training_mode==1:
    while epoch<n_epochs:
            epoch_loss = 0
            num_files = 0
            random.shuffle(train_example_paths)
            model.train(True)
            for example_path in train_example_paths:

                multichannel_audio, target_audio = load_example(example_path)
                # Compute the maximum value across both multichannel and target audio
                
                max_value = torch.max(torch.abs(torch.cat((multichannel_audio, target_audio), dim=1)))
            
                # Normalize the audio signals
                multichannel_audio = multichannel_audio / max_value
                target_audio = target_audio / max_value
                
                pad_size = context
                multichannel_audio = F.pad(multichannel_audio, (pad_size, pad_size), mode='constant', value=0)
                target_audio = F.pad(target_audio, (pad_size, pad_size), mode='constant', value=0)
                
                multichannel_audio = get_overlapping_windows(multichannel_audio, 2**14, 2**13)
                target_audio = get_overlapping_windows(target_audio, 2**14, 2**13)
                running_loss = 0
                
                
                optimizer.zero_grad()
                input_data = multichannel_audio[:,:,:]
                target = target_audio[:,:,context:-context].to(device)
                output = model(input_data.to(device))

                output = output.permute(1, 0, 2).reshape(3, -1)
                target = target.permute(1, 0, 2).reshape(3, -1)
                
                output = output[:,context:]
                target = target[:,context:]
                
                loss = time_weight*snr_loss(output, target.to(device)) + stft_weight*torch.mean(log_error(output, target, n_fft=2**13, win_length=2**13, hop_length=2**13//2))
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                epoch_loss+= running_loss
                num_files +=1
              
            if valid:
          
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
                            

                            val_loss = time_weight*snr_loss(output, target.to(device)) + stft_weight*torch.mean(log_error(output, target, n_fft=2**13, win_length=2**13, hop_length=2**13//2))
                            validation_loss += val_loss.item()
                            num_files += 1

                        validation_loss /= (num_files*multichannel_audio.shape[0])

            print(f"Epoch {epoch+1}/{n_epochs} Train Loss: {epoch_loss / (num_files)}")
            if valid:
                print(f"Validation Loss: {validation_loss}")
            epoch += 1


    # Save final checkpoint after all epochs
    checkpoint = {
        'epoch': n_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint, f"checkpoint_{dataset_path}.pth")

if training_modeand checkpoint:
    model.load_state_dict(torch.load(f"checkpoint_{dataset_path}.pth")["model_state_dict"])
model.eval()


test_loss = 0
j=0
test_example_paths = get_training_example_paths(test_dataset_path)
random.shuffle(test_example_paths)

outputs = []
targets = []

for example_path in test_example_paths:
        multichannel_audio, target_audio = load_example(example_path)
        pad_size = context

        multichannel_audio = F.pad(multichannel_audio, (pad_size, pad_size), mode='constant', value=0)   
        target_audio = F.pad(target_audio, (pad_size, pad_size), mode='constant', value=0)

        multichannel_audio = get_overlapping_windows(multichannel_audio, 2**14, 2**13)
        target_audio = get_overlapping_windows(target_audio, 2**14, 2**13)

        input_data = multichannel_audio[:,:,:]
        target = target_audio[:,:,context:-context]

        output = model(input_data.to(device))

        output = output.permute(1, 0, 2).reshape(3, -1)
        target = target.permute(1, 0, 2).reshape(3, -1)

        outputs.append(output)
        targets.append(target) 

        error = log_error(output.to(device), target.to(device), n_fft=2**13, win_length=2**13, hop_length=2**13//2)
            
        if j==0:
            previous_error = error

        else:
            previous_error = torch.cat((error, previous_error), dim=-1)

        j+=1

error = torch.mean(previous_error, dim=-1)
error = error.detach().cpu().numpy()




