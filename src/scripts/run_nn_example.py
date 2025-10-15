"""Trainer entrypoint for NN-based models (FUSNet / LAENet / SCONet wrappers)."""
import torch
from torch.utils.data import DataLoader
from src.dataset.dataset import RTMDataset
from src.models.laenet import LAENetWrapper
from src.stft.stft_utils import stft_batch, istft_batch
from src.utils.losses import SDRLoss, STFTLogRatioLoss
import Path
from scr.train import Trainer

def run_laenet(device="cuda"):
    n_fft = 2**13
    hop = n_fft // 2
    win = n_fft

    train_ds = RTMDataset("/path/to/train")
    val_ds = RTMDataset("/path/to/val")

    model = LAENetWrapper(
        num_frequencies=n_fft//2+1,
        num_channels=7,
        lstm_hidden_size=7,
        lstm_layers=1,
        num_output_channels=3,
        n_fft=n_fft,
        hop_length=hop
    ).to(device)

    trainer = Trainer(
        estimator=model,
        loss_fns={"mse": torch.nn.MSELoss(), "stft_log_ratio_error": STFTLogRatioLoss(), "sdr_loss": SDRLoss()},
        lr=1e-3,
        epochs=100,
        batch_size=2,
        device=device
    )

    save_dir = Path("checkpoints/laenet")
    ckpt_path = trainer.fit(train_ds, val_ds, save_dir)
    print(f"Training complete. Best model saved at {ckpt_path}")