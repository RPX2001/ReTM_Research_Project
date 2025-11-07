from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn as nn

from . import metrics
from src.utils.retm import block_to_complex


class SDRLoss(nn.Module):
    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return metrics.sdr(output, target)


class STFTLogRatioLoss(nn.Module):
    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return metrics.stft_log_ratio(output, target)


class ComplexCosineLoss(nn.Module):
    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return metrics.complex_cosine(output, target)


class MSELoss(nn.Module):
    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return metrics.mse(output, target)


class LSDLoss(nn.Module):
    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return metrics.lsd(output, target)


class APDLoss(nn.Module):
    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return metrics.apd(output, target)


class RetmLSTFTLoss(nn.Module):
    def __init__(self, output_channels: int, input_channels: int):
        super().__init__()
        self.output_channels = output_channels
        self.input_channels = input_channels

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = block_to_complex(output, self.output_channels, self.input_channels)
        tgt = block_to_complex(target, self.output_channels, self.input_channels)
        total, _ = metrics.lstft_loss_retm(pred, tgt)
        return total


@dataclass
class LossSpec:
    name: str
    fn: nn.Module
    prediction_key: str
    weight: float = 1.0
    target_transform: Optional[Callable[..., torch.Tensor]] = None
