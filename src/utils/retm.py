from typing import Callable
import torch


def covariance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    prod = b.unsqueeze(1).to(torch.complex64) * a.conj().unsqueeze(0).to(torch.complex64)
    return prod.mean(dim=-1)


def retm_from_stft(input_stft: torch.Tensor, target_stft: torch.Tensor) -> torch.Tensor:
    cov_inp_tgt = covariance(target_stft, input_stft).permute(2, 0, 1)   # (F, Cin, Cout)
    cov_trg_trg = covariance(target_stft, target_stft).permute(2, 0, 1)  # (F, Cout, Cout)
    pinv = torch.linalg.pinv(cov_inp_tgt)                                # (F, Cout, Cin)
    return torch.matmul(cov_trg_trg, pinv)                               # (F, Cout, Cin)


def batch_retm_from_stft(input_stft: torch.Tensor, target_stft: torch.Tensor) -> torch.Tensor:
    if input_stft.shape[0] != target_stft.shape[0]:
        raise ValueError(
            "Input and target STFT batches must have the same batch size for ReTM estimation."
        )
    blocks = []
    for b in range(input_stft.shape[0]):
        retm_complex = retm_from_stft(input_stft[b], target_stft[b])
        blocks.append(complex_to_block(retm_complex))
    return torch.stack(blocks, dim=0)


def batch_retm_from_signals(
    stft_fn: Callable[[torch.Tensor], torch.Tensor],
    input_signal: torch.Tensor,
    target_signal: torch.Tensor,
) -> torch.Tensor:
    input_stft = stft_fn(input_signal)
    target_stft = stft_fn(target_signal)
    return batch_retm_from_stft(input_stft, target_stft)


def complex_to_block(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert a complex ReTM matrix (..., Cout, Cin) into its real-valued block form
    (..., 2*Cout, 2*Cin) suitable for real-valued einsum operations.
    """
    if not torch.is_complex(matrix):
        raise ValueError("Expected a complex-valued ReTM matrix for conversion.")
    real = matrix.real
    imag = matrix.imag
    top = torch.cat([real, -imag], dim=-1)
    bottom = torch.cat([imag, real], dim=-1)
    return torch.cat([top, bottom], dim=-2)


def block_to_complex(block: torch.Tensor, output_channels: int, input_channels: int) -> torch.Tensor:
    """
    Convert a real-valued block ReTM of shape (..., 2*Cout, 2*Cin) back to complex form
    (..., Cout, Cin).
    """
    if block.shape[-2] != 2 * output_channels or block.shape[-1] != 2 * input_channels:
        raise ValueError(
            "Block matrix has incompatible shape for the provided channel counts "
            f"(expected (..., {2 * output_channels}, {2 * input_channels}), got {block.shape})."
        )
    top = block[..., :output_channels, :]
    bottom = block[..., output_channels:, :]

    top_left = top[..., :input_channels]
    top_right = top[..., input_channels:]
    bottom_left = bottom[..., :input_channels]
    bottom_right = bottom[..., input_channels:]

    real = 0.5 * (top_left + bottom_right)
    imag = 0.5 * (bottom_left - top_right)
    return torch.complex(real, imag)
