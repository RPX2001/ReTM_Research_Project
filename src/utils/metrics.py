from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from src.utils.retm import block_to_complex


def _flatten_except_batch(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dim() <= 1:
        return tensor.view(1, -1)
    batch = tensor.shape[0]
    return tensor.reshape(batch, -1)


def sdr(pred, target, eps: float = 1e-8, reduce: bool = True):
    noise = target - pred
    sig = torch.sum(_flatten_except_batch(target) ** 2, dim=-1)
    noi = torch.sum(_flatten_except_batch(noise) ** 2, dim=-1) + eps
    values = - 10.0 * torch.log10(sig / noi)
    return values.mean() if reduce else values


def stft_log_ratio(pred, target, eps: float = 1e-10, reduce: bool = True):
    err = torch.abs(pred - target) ** 2
    norm = err / (torch.abs(target) ** 2 + eps)
    values = 10.0 * torch.log10(norm + eps)
    reduced = _flatten_except_batch(values).mean(dim=-1)
    return reduced.mean() if reduce else reduced


def mse(pred, target, reduce: bool = True):
    diff = (pred - target) ** 2
    reduced = _flatten_except_batch(diff).mean(dim=-1)
    reduced = reduced.mean() if reduce else reduced
    return 10.0 * torch.log10(reduced)


def lsd(pred, target, eps: float = 1e-8, reduce: bool = True):
    tgt_mag = torch.abs(target) + eps
    pred_mag = torch.abs(pred) + eps
    diff = 20 * torch.log10(pred_mag) - 20 * torch.log10(tgt_mag)
    diff_sq = _flatten_except_batch(diff ** 2).mean(dim=-1)
    values = torch.sqrt(diff_sq)
    values = values.mean() if reduce else values
    return 10 * torch.log10(values + eps)


def apd(pred, target, reduce: bool = True):
    diff = torch.abs(torch.angle(pred) - torch.angle(target))
    reduced = _flatten_except_batch(diff).mean(dim=-1)
    reduced = reduced.mean() if reduce else reduced
    return 10.0 * torch.log10(reduced + 1e-8)


def complex_cosine(pred, target, eps: float = 1e-8, reduce: bool = True):
    dot = _flatten_except_batch(pred.real * target.real + pred.imag * target.imag).sum(dim=-1)
    pred_norm = torch.sqrt(_flatten_except_batch(pred.real ** 2 + pred.imag ** 2).sum(dim=-1))
    tgt_norm = torch.sqrt(_flatten_except_batch(target.real ** 2 + target.imag ** 2).sum(dim=-1))
    values = 1.0 - (dot / (pred_norm * tgt_norm + eps))
    return values.mean() if reduce else values


def _batched_stft(
    signal: torch.Tensor,
    n_fft: int,
    hop_length: int,
    win_length: int | None = None,
) -> torch.Tensor:
    """Apply STFT to tensors with arbitrary leading dimensions."""
    win_length = win_length or n_fft
    window = torch.hann_window(win_length, device=signal.device)
    if signal.dim() <= 2:
        return torch.stft(
            signal,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            return_complex=True,
        )

    *leading, time = signal.shape
    flat = signal.reshape(-1, time)
    stft = torch.stft(
        flat,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        return_complex=True,
    )
    freq, frames = stft.shape[-2:]
    return stft.reshape(*leading, freq, frames)


def compute_waveform_metrics(
    out_time: torch.Tensor,
    tgt_time: torch.Tensor,
    n_fft: int,
    hop_length: int | None = None,
    eps: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    hop = hop_length or n_fft // 2
    tgt_stft = _batched_stft(tgt_time, n_fft=n_fft, hop_length=hop)
    out_stft = _batched_stft(out_time, n_fft=n_fft, hop_length=hop)
    mse_vals = mse(out_time, tgt_time, reduce=False)
    per_sample = {
        "sdr_db": sdr(out_time, tgt_time, reduce=False),
        "stft_log_ratio": stft_log_ratio(out_stft, tgt_stft, reduce=False),
        "mse_db": mse_vals,
        "lsd_db": lsd(out_stft, tgt_stft, reduce=False),
        "apd_db": apd(out_stft, tgt_stft, reduce=False),
        "complex_cosine_loss": complex_cosine(out_stft, tgt_stft, reduce=False),
    }
    return per_sample


def magnitude_loss_retm(retm_pred: torch.Tensor, retm_true: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mag_pred = torch.abs(retm_pred)
    mag_true = torch.abs(retm_true)
    numerator = torch.norm(mag_pred - mag_true, p=2)
    denominator = torch.norm(mag_true, p=2) + eps
    return numerator / denominator


def phase_loss_retm(retm_pred: torch.Tensor, retm_true: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    phase_pred = torch.angle(retm_pred)
    phase_true = torch.angle(retm_true)
    return torch.mean(torch.abs(phase_pred - phase_true))


def spectral_convergence_retm(retm_pred: torch.Tensor, retm_true: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mag_pred = torch.abs(retm_pred)
    mag_true = torch.abs(retm_true)
    numerator = torch.norm(mag_pred - mag_true, p=2)
    denominator = torch.norm(mag_true, p=2) + eps
    return numerator / denominator


def lstft_loss_retm(retm_pred: torch.Tensor, retm_true: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    l_mag = magnitude_loss_retm(retm_pred, retm_true)
    l_phase = phase_loss_retm(retm_pred, retm_true)
    l_sc = spectral_convergence_retm(retm_pred, retm_true)
    total = l_mag + l_phase + l_sc
    return total, {"magnitude": l_mag, "phase": l_phase, "spectral_convergence": l_sc}


def compute_retm_metrics(
    retm_pred_block: torch.Tensor,
    retm_true_block: torch.Tensor,
    output_channels: int,
    input_channels: int,
) -> Dict[str, torch.Tensor]:
    retm_pred = block_to_complex(retm_pred_block, output_channels, input_channels)
    retm_true = block_to_complex(retm_true_block, output_channels, input_channels)

    metrics: Dict[str, torch.Tensor] = {}
    mse = torch.mean(torch.abs(retm_pred - retm_true) ** 2)
    metrics["retm_mse"] = 10.0 * torch.log10(mse + 1e-12)
    lstft_total, lstft_parts = lstft_loss_retm(retm_pred, retm_true)
    metrics["retm_lstft"] = 10.0 * torch.log10(lstft_total + 1e-12)
    metrics.update({f"retm_lstft/{name}": 10.0 * torch.log10(value + 1e-12) for name, value in lstft_parts.items()})
    return metrics
