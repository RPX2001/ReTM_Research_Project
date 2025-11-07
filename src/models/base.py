from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn

from src.utils.retm import batch_retm_from_signals

@dataclass
class ModelOutput:
    waveform: Optional[torch.Tensor] = None  # (B, C_out, T)
    stft: Optional[torch.Tensor] = None      # (B, F, T, C_out) complex
    retm: Optional[torch.Tensor] = None      # (B, F, 2*C_out, 2*C_in)

    def as_dict(self) -> Dict[str, torch.Tensor]:
        return {k: v for k, v in vars(self).items() if v is not None}

class RTMModel(nn.Module):
    def __init__(self, n_fft: int, hop_length: Optional[int] = None, win_length: Optional[int] = None):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length or n_fft // 2
        self.win_length = win_length or n_fft

    def stft(self, signal: torch.Tensor) -> torch.Tensor:
        """Apply STFT channel-wise while preserving leading dims."""
        window = torch.hann_window(self.win_length, device=signal.device)
        if signal.dim() <= 2:
            return torch.stft(
                signal,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=window,
                return_complex=True,
            )

        *leading, time = signal.shape
        flat = signal.reshape(-1, time)
        stft_flat = torch.stft(
            flat,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            return_complex=True,
        )
        freq, frames = stft_flat.shape[-2:]
        return stft_flat.reshape(*leading, freq, frames)

    def istft(self, spectrum: torch.Tensor, length: Optional[int] = None) -> torch.Tensor:
        """Inverse STFT for tensors whose last dims are (freq, frames)."""
        window = torch.hann_window(self.win_length, device=spectrum.device)
        if spectrum.dim() <= 2:
            return torch.istft(
                spectrum,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=window,
                length=length,
            )

        *leading, freq, frames = spectrum.shape
        flat = spectrum.reshape(-1, freq, frames)
        time = torch.istft(
            flat,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            length=length,
        )
        return time.reshape(*leading, time.shape[-1])

    def forward(self, input_signal: torch.Tensor, target_signal: Optional[torch.Tensor] = None) -> ModelOutput:
        raise NotImplementedError

    def retm_target(self, input_signal: torch.Tensor, target_signal: torch.Tensor) -> torch.Tensor:
        """Compute ReTM targets from raw waveforms via the model STFT helper."""
        return batch_retm_from_signals(self.stft, input_signal, target_signal)
