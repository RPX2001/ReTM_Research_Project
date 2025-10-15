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
from src.dataset.dataset import RTMDataset
from src.train.trainer import Trainer
from src.estimators.closed_form import ClosedFormRTM
from src.models.fusnet import FUSNetWrapper
from src.models.laenet import LAENetWrapper
from src.models.sconet import SCONetWrapper
from src.utils import losses

def parse_args():
    parser = argparse.ArgumentParser(description="RTM Project Entrypoint")
    parser.add_argument("--mode", type=str, choices=["closed_form", "fusnet", "laenet", "sconet"], required=True,
                        help="Which method to run")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root directory for dataset containing mic_*/ subfolders")
    parser.add_argument("--validation_data_root", type=str, required=True,
                        help="Root directory for dataset containing mic_*/ subfolders")
    parser.add_argument("--test_data_root", type=str, required=True,
                        help="Root directory for dataset containing mic_*/ subfolders")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for optimizers")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Where to save models/outputs")
    parser.add_argument("--test", action="store_true",
                    help="Run in test mode instead of training")
    parser.add_argument("--checkpoint", type=str, default=None,
                    help="Path to a trained model checkpoint (.pt) for testing")
    parser.add_argument("--input_mics_numbers", nargs='+', type=int, default = [4,5,6,7], help = "Input mic numbers seperated with space" )
    parser.add_argument("--output_mics_numbers", nargs='+', type=int, default = [1,2,3], help = "Output mic numbers seperated with space" )

    # Model args
    parser.add_argument("--n_fft", type=int, default= 2**13, help= "Number of STFT Frequencies")
    parser.add_argument("--context", type=int, default= 2**13//2, help= "Context length for FUSENet")
    parser.add_argument("--lstm_layers", type=int, default= 1, help= "Number of lstm layers for LAENet model")

    return parser.parse_args()

def main():
    args = parse_args()
    save_path = Path(args.save_dir)
    save_path.mkdir(exist_ok=True, parents=True)

    # Load datasets
    dataset = RTMDataset(args.data_root, input_mics= args.input_mics_numbers, target_mics= args.output_mics_numbers)
    val_dataset = RTMDataset(args.validation_data_root, input_mics= args.input_mics_numbers, target_mics= args.output_mics_numbers)
    test_dataset = RTMDataset(args.test_data_root, input_mics= args.input_mics_numbers, target_mics= args.output_mics_numbers)

    if args.mode == "closed_form":
        estimator = ClosedFormRTM(reg_lambda=1e-2)
        trainer = Trainer(estimator=estimator)

        if args.test:
            print("Running Closed-form Estimator on Test Set...")
            trainer.evaluate(test_dataset, save_dir=save_path)
        else:
            print("Training Closed-form Estimator...")
            trainer.fit(dataset, val_dataset, save_dir=save_path)

    else:
        # Choose neural model
        if args.mode == "fusnet":
            model = FUSNetWrapper(args)
        elif args.mode == "laenet":
            model = LAENetWrapper(args)
        elif args.mode == "sconet":
            model = SCONetWrapper(args)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        loss_fns = {
            "sdr": losses.SDRLoss(),
            "stft": losses.STFTLogRatioLoss()
        }

        trainer = Trainer(
            estimator=model,
            loss_fns=loss_fns,
            lr=args.lr,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=device
        )

        if args.test:
            print(f"Loading checkpoint {args.checkpoint}")
            model.load_state_dict(torch.load(args.checkpoint, map_location=device))
            print("Evaluating on Test Set...")
            trainer.evaluate(test_dataset, save_dir=save_path)
        else:
            print(f"Training {args.mode}...")
            trainer.fit(dataset, val_dataset, save_dir=save_path)

if __name__ == "__main__":
    main()
