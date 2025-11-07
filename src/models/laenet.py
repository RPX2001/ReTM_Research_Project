import torch
import torch.nn as nn
from .base import ModelOutput, RTMModel
from src.utils.retm import block_to_complex

class FrequencyLSTMNet(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, n_fft: int, lstm_layers: int):
        super().__init__()
        total_channels = input_channels + output_channels
        hidden = total_channels
        self.output_channels = output_channels
        self.input_channels = input_channels
        self.n_fft = n_fft
        self.hop_length = n_fft // 2

        self.lstm = nn.LSTM(input_size=2 * total_channels,
                            hidden_size=2 * hidden,
                            num_layers=lstm_layers,
                            batch_first=True,
                            bidirectional=True)
        self.layer_norm = nn.LayerNorm(4 * hidden)

        out_dim = 4 * (output_channels * input_channels)
        self.fc = nn.Sequential(
            nn.Linear(4 * hidden, out_dim * 2),
            nn.ReLU(),
            nn.Linear(out_dim * 2, out_dim * 4),
            nn.ReLU(),
            nn.Linear(out_dim * 4, out_dim),
        )

    def forward(self, stft_concat: torch.Tensor) -> torch.Tensor:
        batch, _, freq, time = stft_concat.shape  # (B, 2*C_tot, F, T)
        lstm_in = stft_concat.permute(0, 2, 3, 1).reshape(batch * freq, time, -1)
        lstm_out, _ = self.lstm(lstm_in)
        fused = self.layer_norm(lstm_out.mean(dim=1))
        fc_out = self.fc(fused).view(batch, freq,
                                     2 * self.output_channels,
                                     2 * self.input_channels)
        return fc_out

class LAENetWrapper(RTMModel):
    def __init__(self, args):
        super().__init__(n_fft=args.n_fft)
        self.input_channels = len(args.input_mics_numbers)
        self.output_channels = len(args.output_mics_numbers)
        self.core = FrequencyLSTMNet(self.input_channels, self.output_channels,
                                     args.n_fft, args.lstm_layers)

    def forward(self, input_signal: torch.Tensor, target_signal: torch.Tensor | None = None) -> ModelOutput:
        batch, _, time = input_signal.shape
        if target_signal is None:
            target_signal = torch.zeros(batch, self.output_channels, time, device=input_signal.device)

        full = torch.cat([target_signal, input_signal], dim=1)  # (B, Cout+Cin, T)
        stft_full = self.stft(full)                             # (B, Cout+Cin, F, T) complex
        stft_ri = torch.view_as_real(stft_full).reshape(batch, -1, stft_full.shape[-2], stft_full.shape[-1])
        retm = self.core(stft_ri)

        input_stft = stft_full[:, self.output_channels:, ...]   # (B, Cin, F, T) complex
        input_ri = torch.view_as_real(input_stft).permute(0, 2, 3, 1, 4).reshape(batch, self.n_fft // 2 + 1, -1, 2 * self.input_channels)
        out_ri = torch.einsum("bfnc,bftc->bftn", retm, input_ri)
        out_stft = (out_ri[..., 0::2] + 1j * out_ri[..., 1::2]).permute(0, 3, 1, 2)  # (B, Cout, F, T)

        hann = torch.hann_window(self.win_length, device=input_signal.device)
        out_time = torch.istft(out_stft.reshape(-1, out_stft.shape[2], out_stft.shape[3]),
                               n_fft=self.n_fft, hop_length=self.hop_length,
                               win_length=self.win_length, window=hann,
                               length=time)
        out_time = out_time.view(batch, self.output_channels, -1)

        return ModelOutput(
            waveform=out_time,
            stft=out_stft.permute(0, 2, 3, 1),  # (B, F, T, Cout)
            retm=retm
        )

    def retm_target(self, input_signal: torch.Tensor, target_signal: torch.Tensor) -> torch.Tensor:
        return super().retm_target(input_signal, target_signal)

    def retm_block_to_complex(self, retm_block: torch.Tensor) -> torch.Tensor:
        return block_to_complex(retm_block, self.output_channels, self.input_channels)
