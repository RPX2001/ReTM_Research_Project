import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import torchaudio.functional
import torchaudio.transforms as T
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d  # For smoothing

# Define the dataset path
dataset_path = "train_dataset_C_10_denoise"
valid_dataset_path = "valid_dataset_C_10"


class LogRatioLoss(nn.Module):
    """
    Custom loss function that computes the mean log-ratio error
    between predicted and target magnitudes.
    """
    def __init__(self):
        super(LogRatioLoss, self).__init__()

    def forward(self, predicted, target):
        """
        Computes the mean log-ratio error.

        Parameters:
        - predicted: Tensor of predicted magnitudes, shape (freq, time)
        - target: Tensor of target magnitudes, shape (freq, time)

        Returns:
        - Mean log-ratio error (scalar)
        """
        # Separate real and imaginary parts for predicted and target
        predicted_img = predicted[5:10, :,:]
        predicted_real = predicted[0:5, :,:]
        target_img = target[5:10, :,:]
        target_real = target[0:5, :,:]

        # Convert to complex numbers
        predicted_complex = predicted_real + 1j * predicted_img
        target_complex = target_real + 1j * target_img

        # Compute the squared error
        squared_error = torch.abs(predicted_complex - target_complex) ** 2

        # Normalize by the square of the target
        normalized_error = squared_error / (torch.abs(target_complex) ** 2 + 1e-10)

        # Take log10 and then multiply by 10
        log_error = 10 * torch.log10(normalized_error + 1)  # Numerical stability
        
        # Compute the mean across time (axis=-1) and return the final mean error
        mean_log_ratio_error = torch.mean(log_error, dim=-1).mean()

        return mean_log_ratio_error

def log_ratio_error(predicted, target):
    """
    Computes the mean of the log-ratio error as specified.
    
    Parameters:
    - predicted: Tensor of predicted magnitudes, shape (freq, time)
    - target: Tensor of target magnitudes, shape (freq, time)
    
    Returns:
    - Mean log-ratio error
    """
    
    predicted_img = predicted[5:10, :,:]
    predicted_real = predicted[0:5, :,:]
    target_img = target[5:10, :,:]
    target_real = target[0:5, :,:]
    predicted = predicted_real + 1j * predicted_img
    target = target_real + 1j * target_img
    # Compute the squared error
    squared_error = (abs(predicted - target)) ** 2
    
    # Normalize by the square of the target
    normalized_error = squared_error / (abs(target)** 2 + 1e-10)
    
    # Take log10 and then multiply by 10
    log_error = 10 * torch.log10(normalized_error + 1e-10)  # Adding a small value for numerical stability
    print("shape is ", log_error.shape)
    # Compute the mean across time (axis=1) and return the final mean error
    mean_log_ratio_error = torch.mean(log_error, dim=2)
    
    
    return mean_log_ratio_error


# Function to load training examples
def load_training_examples(dataset_path):   
    # Get a list of all subfolders in the dataset folder
    example_paths = [os.path.join(dataset_path, folder) for folder in os.listdir(dataset_path)]
    # Iterate over each example folder
    for example_path in example_paths:
        recordings = []
        target_recordings = []
        
        # Load normal recordings (mic_1, mic_2, mic_3, mic_4)
        for i in range(6,13 ):
            filepath = os.path.join(example_path, f"mic_{i}.wav")
            if os.path.isfile(filepath):
                waveform, _ = torchaudio.load(filepath)
                recordings.append(waveform)
        
        # Load target recordings (mic_5, mic_6)
        for i in range(1, 6):
            filepath = os.path.join(example_path, f"mic_{i}.wav")
            if os.path.isfile(filepath):
                waveform, _ = torchaudio.load(filepath)
                target_recordings.append(waveform)
        
        # Ensure data is tensorized and truncated to minimum length
        if recordings and target_recordings:
            multichannel_audio = torch.stack(recordings, dim=0).squeeze(1)
            target_audio = torch.stack(target_recordings, dim=0).squeeze(1)
            min_length = min(multichannel_audio.shape[1], target_audio.shape[1])
            multichannel_audio, target_audio = multichannel_audio[:, :min_length], target_audio[:, :min_length]
            
            yield multichannel_audio, target_audio

win_length = 2**13
n_fft = win_length

class DepthwiseConvModel(nn.Module):
    def __init__(self, channels, n_output_channels):
        super(DepthwiseConvModel, self).__init__()
        self.stft_transform = T.Spectrogram(n_fft=n_fft, hop_length=win_length//2, win_length=win_length, power=None) 
        self.depthwise_conv = nn.Conv2d(
            in_channels=channels,
            out_channels=n_output_channels * channels,
            kernel_size=(2*7, 1),
            stride=1,
            padding=0,
            groups=channels,
            bias=False    
        )

    def forward(self, audio_signal):
        # Compute STFT
        stft_result = self.stft_transform(audio_signal)  # Shape: (num_channels, freq, time)

        stft_result = torch.cat((stft_result[:,:,:,0],stft_result[:,:,:,1]),dim=0)

        stft_result = torch.swapaxes(stft_result, 0, 1)

        output = self.depthwise_conv(stft_result)

        output = output.squeeze()
        
        output = self.divide_and_stack(output, n_output_channels, axis=0)
        
        output = torch.swapaxes(output, 0, 1)
        
 
        return output

    @staticmethod
    def divide_and_stack(tensor, num_splits, axis):
        divided_tensors = torch.split(tensor, num_splits, dim=axis)
        
        return torch.stack(divided_tensors, dim=0)

# Initialize the model
channels =n_fft//2+1 # Set based on your data
n_output_channels = 5*2
model = DepthwiseConvModel(channels, n_output_channels)

# Define optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.01)

criterion = nn.MSELoss()
# criterion = LogRatioLoss()


#loss_fn = LogRatioLoss()
mse_loss_history = []
validation_loss_history = []
log_ratio_error_history = []
min_log_ratio_error = float('inf')

print(model)
# Training loop
epochs = 100#Set according to your needs
print()
valid = 0

for epoch in range(epochs):
    for multichannel_audio, target_audio in load_training_examples(dataset_path):
        model.train()
        
        # Compute the maximum value across both multichannel and target audio
        max_value = torch.max(torch.abs(torch.cat((multichannel_audio, target_audio), dim=0)))
        
        # Normalize the audio signals
        multichannel_audio = multichannel_audio / max_value
        target_audio = target_audio / max_value
            
        # Forward pass
        output = model(multichannel_audio)
        target_output = model.stft_transform(target_audio)
        target_output = torch.cat((target_output[:,:,:,0],target_output[:,:,:,1]),dim=0)

        loss = criterion(output, target_output)

        
        # Backpropagation and optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    
    if valid==1:
        model.eval()
        validation_loss = 0
        with torch.no_grad():
            for multichannel_audio, target_audio in load_training_examples(valid_dataset_path):  # Use a separate validation dataset here
                
                max_value = torch.max(torch.abs(torch.cat((multichannel_audio, target_audio), dim=0)))
            
                # Normalize the audio signals
                multichannel_audio = multichannel_audio / max_value
                target_audio = target_audio / max_value
                
                output = model(multichannel_audio)
                target_output = model.stft_transform(target_audio)
                target_output = torch.cat((target_output[:, :, :, 0], target_output[:, :, :, 1]), dim=0)

                val_loss = criterion(output, target_output)
                validation_loss += val_loss.item()

        validation_loss /= len(os.listdir(valid_dataset_path))
        validation_loss_history.append(validation_loss)
            
            
    print(f"Epoch [{epoch+1}/{epochs}] Complete. Loss: {loss.item():.4f}")    
    if valid==1:
        print(f"Validation Loss: {validation_loss:.4f}")
    mse_loss_history.append(loss.item())

    
print("Training complete.")

test_dataset_path = "test_dataset_C_10_denoise"

model.eval()

test_loss = 0

i = 0

for multichannel_audio, target_audio in load_training_examples(test_dataset_path):
    max_value = torch.max(torch.abs(torch.cat((multichannel_audio, target_audio), dim=0)))
        
    # Normalize the audio signals
    multichannel_audio = multichannel_audio / max_value
    target_audio = target_audio / max_value
    target_output = model.stft_transform(target_audio)
    target_output = torch.cat((target_output[:,:,:,0],target_output[:,:,:,1]),dim=0) ##[channels, freq, time]
    output = model(multichannel_audio)
    
    if i==0:
        target_test_output = target_output
        test_output = output
    else:
        target_test_output = torch.cat((target_test_output, target_output), dim=2)
        test_output = torch.cat((test_output, output), dim=2)
    i+=1
    loss = criterion(output, target_output)
    test_loss += loss.item()

test_loss /= len(os.listdir(test_dataset_path))

print(f"Test Loss: {test_loss:.4f}") 

denoising_file = "rec_scen_C.wav"

# Load the multi-channel file
waveform, sample_rate = torchaudio.load("rec_scen_C.wav")  # waveform shape: (channels, time)

# For example, extract channels 0 and 2
output_indices = [0, 1, 2, 3, 4]
input_indices = [5, 6, 7, 8, 9, 10, 11]
target_recordings = waveform[output_indices, :]
multichannel_audio = waveform[input_indices, :]


output = model(multichannel_audio)
output_img = output[5:10, :,:]
output_real = output[0:5, :,:]
output = output_real + 1j * output_img
target_output = model.stft_transform(target_recordings)
denoised_stft = torch.view_as_complex(target_output) - output
denoised_sig = torch.istft(
    denoised_stft, 
    n_fft=n_fft,
    hop_length=win_length//2,
    win_length=win_length
)

torchaudio.save("denoised_sig_scenario_C.wav", denoised_sig.detach(), sample_rate)

plt.figure(figsize=(6, 4))
plt.plot(target_recordings[0,:].detach().numpy(), label="Original")
plt.plot(denoised_sig[0,:].detach().numpy(), label="Denoised")
plt.xlabel("Time (sample)")
plt.ylabel("Magnitude")
plt.title("Denoised signal")
plt.grid(True)
plt.legend()
plt.show()



