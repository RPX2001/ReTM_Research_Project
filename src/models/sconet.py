import torch
import torch.nn as nn
from .base import ModelOutput, RTMModel
from src.utils.retm import batch_retm_from_stft

class DepthwiseConvModel(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, n_fft: int):
        super().__init__()
        freq_bins = n_fft // 2 + 1
        self.depthwise = nn.Conv2d(
            in_channels=freq_bins,
            out_channels=freq_bins * (2 * output_channels),
            kernel_size=(2 * input_channels, 1),
            stride=1,
            padding=0,
            groups=freq_bins,
            bias=False,
        )
        self.output_channels = output_channels
        self.freq_bins = freq_bins

    def forward(self, stft_real_imag: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(stft_real_imag).squeeze(2)
        splits = torch.split(x, self.output_channels * 2, dim=1)
        stacked = torch.stack(splits, dim=2)  # (B, Cout*2, F, T)
        return stacked

class SCONetWrapper(RTMModel):
    def __init__(self, args):
        super().__init__(n_fft=args.n_fft)
        self.input_channels = len(args.input_mics_numbers)
        self.output_channels = len(args.output_mics_numbers)
        self.model = DepthwiseConvModel(self.input_channels, self.output_channels, args.n_fft)
        profile = (getattr(args, "loss_profile", "retm") or "retm").lower()
        self._emit_retm = profile in {"retm", "hybrid"}

    def forward(self, input_signal: torch.Tensor, target_signal: torch.Tensor | None = None) -> ModelOutput:
        stft = self.stft(input_signal)  # (B, Cin, F, T)
        ri = torch.view_as_real(stft).permute(0, 2, 1, 3, 4).reshape(stft.shape[0], -1, self.input_channels * 2, stft.shape[-1])
        conv_out = self.model(ri)
        real = conv_out[:, :self.output_channels, ...]
        imag = conv_out[:, self.output_channels:, ...]
        out_stft_complex = torch.complex(real, imag)
        retm_block = None
        if self._emit_retm:
            retm_block = batch_retm_from_stft(stft, out_stft_complex)
        out_wave = self.istft(out_stft_complex, length=input_signal.shape[-1])
        out_stft = out_stft_complex.permute(0, 2, 3, 1)
        return ModelOutput(waveform=out_wave, stft=out_stft, retm=retm_block)
