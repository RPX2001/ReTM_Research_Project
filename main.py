#!/usr/bin/env python3
"""
Main entrypoint for RTM project.
Choose between closed-form RTM estimation or NN-based training.
"""

import argparse
import torch
from pathlib import Path

# Import local modules
from src.io import readers
from src.stft import stft_utils
from src.dataset.dataset import MicDataset
from src.train.trainer import Trainer
from src.estimators.closed_form import RidgeRTMEstimator
from src.models.fusnet import FUSNetAdapter
from src.models.laenet import LAENetAdapter
from src.models.sconet import SCONetAdapter
from src.utils import losses

def parse_args():
    parser = argparse.ArgumentParser(description="RTM Project Entrypoint")
    parser.add_argument("--mode", type=str, choices=["closed_form", "fusnet", "laenet", "sconet"], required=True,
                        help="Which method to run")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root directory for dataset containing mic_*/ subfolders")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for optimizers")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Where to save models/outputs")
    return parser.parse_args()

def main():
    args = parse_args()
    save_path = Path(args.save_dir)
    save_path.mkdir(exist_ok=True, parents=True)

    # Dataset
    dataset = MicDataset(args.data_root, split="train")
    val_dataset = MicDataset(args.data_root, split="val")

    if args.mode == "closed_form":
        # ============ CLOSED-FORM RIDGE RTM ===============
        print("Running Closed-form Ridge RTM Estimator...")
        estimator = RidgeRTMEstimator(reg_lambda=1e-2)
        trainer = Trainer(estimator=estimator)
        trainer.fit(dataset, val_dataset, save_dir=save_path)

    else:
        # Pick model
        if args.mode == "fusnet":
            model = FUSNetAdapter()
        elif args.mode == "laenet":
            model = LAENetAdapter()
        elif args.mode == "sconet":
            model = SCONetAdapter()
        else:
            raise ValueError("Unknown mode")

        # Loss functions
        loss_fns = {
            "snr": losses.SNRLoss(),
            "stft": losses.STFTLogRatioLoss()
        }

        # ============ TRAIN NEURAL NET ==================
        print(f"Training Neural Model: {args.mode}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        trainer = Trainer(estimator=model,
                          loss_fns=loss_fns,
                          lr=args.lr,
                          epochs=args.epochs,
                          batch_size=args.batch_size,
                          device=device)
        trainer.fit(dataset, val_dataset, save_dir=save_path)


if __name__ == "__main__":
    main()
