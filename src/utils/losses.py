import torch
import torch.nn as nn

class SDRLoss(nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        noise = target - output
        signal_power = torch.sum(target**2, dim=1)
        noise_power = torch.sum(noise**2, dim=1) + self.eps
        snr = 10 * torch.log10(signal_power / noise_power)
        return -snr.mean()


class STFTLogRatioLoss(nn.Module):
    def __init__(self, eps: float = 1e-10):
        super().__init__()
        self.eps = eps

    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        sq_err = (torch.abs(predicted - target))**2
        norm = sq_err / (torch.abs(target)**2 + self.eps)
        log_err = 10 * torch.log10(norm + self.eps)
        return torch.mean(log_err)