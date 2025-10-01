import torch

def snr_loss(output: torch.Tensor, target: torch.Tensor, eps: float=1e-8) -> torch.Tensor:
    noise = target - output
    signal_power = torch.sum(target**2, dim=1)
    noise_power = torch.sum(noise**2, dim=1) + eps
    snr = 10 * torch.log10(signal_power / noise_power)
    return -snr.mean()


def stft_log_ratio_error(predicted, target, eps:float=1e-10):
    # predicted/target are complex tensors
    sq_err = (torch.abs(predicted - target))**2
    norm = sq_err / (torch.abs(target)**2 + eps)
    log_err = 10 * torch.log10(norm + eps)
    return torch.mean(log_err)