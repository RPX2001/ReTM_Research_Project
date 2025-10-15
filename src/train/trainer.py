import torch
from torch.utils.data import DataLoader
from src.dataset.dataset import RTMDataset
from src.utils.losses import snr_loss, stft_log_ratio_error
from src.stft.stft_utils import stft_batch, istft_batch
from src.estimators.closed_form import ClosedFormRTM
from pathlib import Path
import os


def train_closed_form(dataset_root, n_fft=2**13, hop=None, win=None, device='cpu'):
    if hop is None: hop = n_fft//2
    if win is None: win = n_fft
    ds = RTMDataset(dataset_root)
    dl = DataLoader(ds, batch_size=4, shuffle=True, num_workers=4)
    estimator = ClosedFormRTM(reg=1e-3)
    estimator.to = lambda d: None
    for inp, tgt in dl:
        # inp: (batch, in_ch, samples)
        inp = inp.to(device)
        tgt = tgt.to(device)
        inp_stft = stft_batch(inp, n_fft, hop, win, device=device, return_complex=True)  # (batch,freq,time,chan)
        tgt_stft = stft_batch(tgt, n_fft, hop, win, device=device, return_complex=True)
        rtm = estimator.fit(inp_stft, tgt_stft)
        out = estimator.predict(inp_stft)
        wav_out = istft_batch(out, n_fft, hop, win, device=device)
        wav_tgt = istft_batch(tgt_stft, n_fft, hop, win, device=device)
        loss_time = snr_loss(wav_out.view(wav_out.shape[0]*wav_out.shape[1], -1), wav_tgt.view(wav_tgt.shape[0]*wav_tgt.shape[1], -1))
        loss_stft = stft_log_ratio_error(out, tgt_stft)
        loss = loss_time + loss_stft
        print('batch loss', loss.item())

class Trainer:
    def __init__(self, estimator, loss_fns=None, lr=1e-3, epochs=10, batch_size=4, device="cpu"):
        self.estimator = estimator
        self.loss_fns = loss_fns or {}
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device

        # Only create optimizer if estimator is a NN
        if hasattr(estimator, "parameters"):
            self.optimizer = torch.optim.Adam(estimator.parameters(), lr=lr)
        else:
            self.optimizer = None

    def fit(self, train_dataset, val_dataset, save_dir: Path):
        save_dir.mkdir(exist_ok=True, parents=True)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        best_val_loss = float("inf")
        best_ckpt_path = save_dir / "best_model.pt"

        for epoch in range(1, self.epochs + 1):
            self.estimator.train()
            train_losses = {k: 0.0 for k in self.loss_fns}

            for batch in train_loader:
                x, y = batch["input"].to(self.device), batch["target"].to(self.device)

                if self.optimizer:
                    self.optimizer.zero_grad()

                preds = self.estimator(x)

                loss_total = 0.0
                for name, fn in self.loss_fns.items():
                    l = fn(preds, y)
                    train_losses[name] += l.item()
                    loss_total += l

                if self.optimizer:
                    loss_total.backward()
                    self.optimizer.step()

            # average train loss
            n_train_batches = len(train_loader)
            train_losses = {k: v / n_train_batches for k, v in train_losses.items()}

            # validation
            val_losses = self.evaluate(val_loader, save_dir, log=False)

            # save best model checkpoint
            val_loss_total = sum(val_losses.values())
            if val_loss_total < best_val_loss and self.optimizer is not None:
                best_val_loss = val_loss_total
                torch.save(self.estimator.state_dict(), best_ckpt_path)
                print(f"[EPOCH {epoch}] ✅ Saved best checkpoint: {best_ckpt_path}")

            print(f"[EPOCH {epoch}] Train: {train_losses} | Val: {val_losses}")

        return best_ckpt_path

    def evaluate(self, dataset_or_loader, save_dir: Path, log=True):
        """Evaluate on a dataset or pre-built dataloader"""
        if hasattr(dataset_or_loader, "__getitem__"):  # Dataset
            dataloader = DataLoader(dataset_or_loader, batch_size=self.batch_size, shuffle=False)
        else:
            dataloader = dataset_or_loader

        self.estimator.eval()
        total_loss = {k: 0.0 for k in self.loss_fns}

        with torch.no_grad():
            for batch in dataloader:
                x, y = batch["input"].to(self.device), batch["target"].to(self.device)
                preds = self.estimator(x)

                for name, fn in self.loss_fns.items():
                    total_loss[name] += fn(preds, y).item()

        # average losses
        n_batches = len(dataloader)
        results = {k: v / n_batches for k, v in total_loss.items()}

        if log:
            print(f"[EVAL] Results: {results}")
            (save_dir / "results.txt").write_text(str(results))

        return results