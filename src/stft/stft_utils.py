import torch

def stft_batch(x: torch.Tensor, n_fft:int, hop:int, win_length:int, device=None, return_complex=True):
    # x: (batch, channels, samples) or (channels, samples)
    single = False
    if x.dim()==2:
        single=True
        x = x.unsqueeze(0)
    batch, ch, samples = x.shape
    stfts = []
    for c in range(ch):
        s = torch.stft(x[:,c,:], n_fft=n_fft, hop_length=hop, win_length=win_length, window=torch.hann_window(n_fft).to(device), return_complex=return_complex)
        # s: (batch, freq, time) or (batch, freq, time, 2)
        stfts.append(s)
    # stack channels into last dim
    if return_complex:
        out = torch.stack(stfts, dim=-1) # (batch,freq,time,channels)
    else:
        out = torch.stack(stfts, dim=-1) # (batch,freq,time,channels,2)
    if single:
        out = out.squeeze(0)
    return out

def istft_batch(X: torch.Tensor, n_fft:int, hop:int, win_length:int, device=None):
    # X: (batch, freq, time, channels) complex
    single=False
    if X.dim()==3:
        single=True
        X = X.unsqueeze(0)
    batch = X.shape[0]
    outs = []
    for b in range(batch):
        # reconstruct per channel and stack
        ch_signals = []
        for c in range(X.shape[-1]):
            sig = torch.istft(X[b,:,:,c], n_fft=n_fft, hop_length=hop, win_length=win_length, window=torch.hann_window(n_fft).to(device))
            ch_signals.append(sig)
        outs.append(torch.stack(ch_signals, dim=0))
    out = torch.stack(outs, dim=0)
    if single:
        return out.squeeze(0)
    return out