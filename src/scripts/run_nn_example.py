"""Trainer entrypoint for NN-based models (FUSNet / LAENet / SCONet wrappers)."""
import torch
from torch.utils.data import DataLoader
from src.dataset.dataset import RTMDataset
from src.models.laenet import LAENetWrapper
from src.stft.stft_utils import stft_batch, istft_batch
from src.utils.losses import snr_loss, stft_log_ratio_error


def train_laenet(device='cuda'):
    n_fft=2**13
    hop = n_fft//2
    win = n_fft
    ds = RTMDataset('/home/lathika/ReTM_Workspace/Recordings/Splited_data/Channels_7/train')
    dl = DataLoader(ds, batch_size=2, shuffle=True)
    model = LAENetWrapper(num_frequencies=n_fft//2+1, num_channels=7, lstm_hidden_size=7, lstm_layers=1, num_output_channels=3, n_fft=n_fft, hop_length=hop)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(100):
        for inp, tgt in dl:
            inp = inp.to(device)
            tgt = tgt.to(device)
            # model outputs an RTM-like tensor: (batch, freq, 2*out_ch, 2*other)
            rtm_pred = model(inp)
            # convert inputs to stft
            inp_stft = stft_batch(inp, n_fft, hop, win, device=device, return_complex=True)
            tgt_stft = stft_batch(tgt, n_fft, hop, win, device=device, return_complex=True)
            # adapter: reshape rtm_pred into complex rtm and apply
            # -----------------
            # TODO: implement adapter depending on model output format
            # -----------------
            # compute loss and step
            # placeholder loss
            loss = torch.tensor(0.0, device=device)
            opt.zero_grad()
            loss.backward()
            opt.step()
        print('epoch', epoch)