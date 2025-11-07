import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import ModelOutput, RTMModel
from src.utils.retm import batch_retm_from_stft

class MultiChannelConv(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, kernel_size: int):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Conv1d(1, output_channels, kernel_size, bias=False)
            for _ in range(input_channels)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = [layer(x[:, i:i+1, :]) for i, layer in enumerate(self.layers)]
        return torch.stack(outs, dim=0).sum(dim=0)

class FUSNetWrapper(RTMModel):
    def __init__(self, args):
        super().__init__(n_fft=args.n_fft)
        self.context = args.context
        self.input_channels = len(args.input_mics_numbers)
        self.output_channels = len(args.output_mics_numbers)
        kernel = 2 * self.context + 1
        self.model = MultiChannelConv(self.input_channels, self.output_channels, kernel)
        profile = (getattr(args, "loss_profile", "retm") or "retm").lower()
        self._emit_retm = profile in {"retm", "hybrid"}

    def forward(self, input_signal: torch.Tensor, target_signal: torch.Tensor | None = None) -> ModelOutput:
        pad = self.context
        x = F.pad(input_signal, (pad, pad), mode="constant", value=0.0)
        out = self.model(x)
        target_len = input_signal.shape[-1]
        current_len = out.shape[-1]
        if current_len > target_len:
            start = (current_len - target_len) // 2
            out_wave = out[..., start:start + target_len]
        elif current_len < target_len:
            pad_total = target_len - current_len
            out_wave = F.pad(out, (0, pad_total))
        else:
            out_wave = out
        out_stft = self.stft(out_wave)
        retm_block = None
        if self._emit_retm:
            input_stft = self.stft(input_signal)
            retm_block = batch_retm_from_stft(input_stft, out_stft)
        return ModelOutput(
            waveform=out_wave,
            stft=out_stft.permute(0, 2, 3, 1),
            retm=retm_block,
        )
