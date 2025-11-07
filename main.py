#!/usr/bin/env python3
"""
Main entrypoint for RTM project.
Choose between closed-form RTM estimation or NN-based training.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

import torch
from torch.utils.data import DataLoader

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

# Import local modules
from src.dataset.dataset import RTMDataset
from src.estimators.closed_form import ClosedFormRTM
from src.models.base import ModelOutput, RTMModel
from src.models.fusnet import FUSNetWrapper
from src.models.laenet import LAENetWrapper
from src.models.sconet import SCONetWrapper
from src.stft import stft_utils
from src.train.trainer import Trainer
from src.utils.losses import (
    APDLoss,
    ComplexCosineLoss,
    LSDLoss,
    LossSpec,
    MSELoss,
    RetmLSTFTLoss,
    SDRLoss,
    STFTLogRatioLoss,
)
from src.utils.metrics import compute_retm_metrics, compute_waveform_metrics

DEFAULT_CONFIG: Dict[str, Any] = {
    "epochs": 10,
    "batch_size": 4,
    "lr": 1e-3,
    "save_dir": "checkpoints",
    "test": False,
    "checkpoint": None,
    "input_mics_numbers": [4, 5, 6, 7],
    "output_mics_numbers": [1, 2, 3],
    "n_fft": 2**13,
    "context": (2**13) // 2,
    "lstm_layers": 1,
    "grad_clip": None,
    "use_amp": True,
    "loss_profile": "retm",
    "time_weight": 1.0,
    "stft_weight": 1.0,
    "retm_mse_weight": 1.0,
    "retm_lstft_weight": 1.0,
    "cs_weight": 1.0,
}

DEFAULT_METRICS: Dict[str, List[str]] = {
    "waveforms": ["sdr_db", "stft_log_ratio", "mse_db", "lsd_db", "apd_db", "complex_cosine"],
    "retm": ["mse", "lstft"],
}

DEFAULT_WANDB: Dict[str, Any] = {
    "enabled": False,
    "project": None,
    "entity": None,
    "run_name": None,
}

WAVEFORM_METRIC_KEYS = set(DEFAULT_METRICS["waveforms"])
RETM_METRIC_MAP = {"mse": "retm_mse", "lstft": "retm_lstft"}


def _closed_form_eval_split(dataset, args, batch_size: int) -> tuple[float, float]:
    if len(dataset) == 0:
        return float("nan"), float("nan")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    estimator = ClosedFormRTM(reg=1e-2)
    sdr_loss = SDRLoss()
    hop = args.n_fft // 2
    total_loss = 0.0
    total_examples = 0

    for batch in loader:
        inputs = batch["input"]
        targets = batch["target"]

        src_stft = stft_utils.stft_batch(inputs, args.n_fft, hop, args.n_fft, device=inputs.device)
        tgt_stft = stft_utils.stft_batch(targets, args.n_fft, hop, args.n_fft, device=targets.device)

        estimator.fit(src_stft, tgt_stft)
        pred_stft = estimator.predict(src_stft)
        pred_wave = stft_utils.istft_batch(pred_stft, args.n_fft, hop, args.n_fft, device=targets.device)

        min_len = min(pred_wave.shape[-1], targets.shape[-1])
        pred_wave = pred_wave[..., :min_len]
        targets = targets[..., :min_len]

        loss_val = sdr_loss(pred_wave, targets)
        batch_size_curr = inputs.size(0)
        total_loss += loss_val.item() * batch_size_curr
        total_examples += batch_size_curr

    avg_loss = total_loss / total_examples if total_examples else float("nan")
    avg_sdr_db = -avg_loss if avg_loss == avg_loss else float("nan")
    return avg_loss, avg_sdr_db


def _run_closed_form(args, dataset, val_dataset, test_dataset, save_path: Path):
    splits = [("train", dataset), ("val", val_dataset)]
    if args.test:
        splits = [("test", test_dataset)]

    for split_name, split_dataset in splits:
        avg_loss, avg_sdr = _closed_form_eval_split(split_dataset, args, args.batch_size)
        metric_text = (
            f"{split_name} average SDR loss: {avg_loss:.6f}\n"
            f"{split_name} average SDR (dB): {avg_sdr:.6f}\n"
        )
        (save_path / f"{split_name}_closed_form.txt").write_text(metric_text)
        print(metric_text.strip())


def parse_args():
    parser = argparse.ArgumentParser(description="RTM Project Entrypoint")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML configuration file (default: config.yaml)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["closed_form", "fusnet", "laenet", "sconet"],
        help="Override model mode defined in the config file",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Override config to run in test/evaluation mode",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Override checkpoint path when running in test mode",
    )
    return parser.parse_args()


def _load_config(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if yaml is None:
        raise RuntimeError("PyYAML is required to load configuration files.")
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError("Configuration file must define a mapping at the top level.")
    return data


def _merge_defaults(config: Dict[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {**DEFAULT_CONFIG, **{k: v for k, v in config.items() if k not in {"metrics", "wandb"}}}

    metrics_cfg = DEFAULT_METRICS | config.get("metrics", {})
    merged["metrics"] = {
        "waveforms": list(metrics_cfg.get("waveforms", [])),
        "retm": list(metrics_cfg.get("retm", [])),
    }

    wandb_cfg = {**DEFAULT_WANDB, **config.get("wandb", {})}
    merged["wandb"] = wandb_cfg
    return merged


def _build_waveform_collector(
    requested: Iterable[str],
    n_fft: int,
    hop_length: int,
) -> Optional[Callable[[ModelOutput, torch.Tensor, torch.Tensor], Dict[str, float]]]:
    active = [name for name in requested if name in WAVEFORM_METRIC_KEYS]
    if not active:
        return None

    def collector(output: ModelOutput, _inputs: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        if output.waveform is None:
            return {}
        metric_values = compute_waveform_metrics(output.waveform.detach(), target.detach(), n_fft=n_fft, hop_length=hop_length)
        return {name: float(metric_values[name].mean().item()) for name in active if name in metric_values}

    def per_sample(output: ModelOutput, _inputs: torch.Tensor, target: torch.Tensor) -> List[Dict[str, float]]:
        if output.waveform is None:
            return []
        metric_values = compute_waveform_metrics(output.waveform.detach(), target.detach(), n_fft=n_fft, hop_length=hop_length)
        batch = output.waveform.shape[0]
        records = [dict() for _ in range(batch)]
        for name, values in metric_values.items():
            for idx in range(min(batch, values.shape[0])):
                records[idx][name] = float(values[idx].item())
        return records

    collector.per_sample = per_sample  # type: ignore[attr-defined]
    return collector


def _build_retm_collector(
    requested: Iterable[str],
    model: RTMModel,
) -> Optional[Callable[[ModelOutput, torch.Tensor, torch.Tensor], Dict[str, float]]]:
    active = [RETM_METRIC_MAP[name] for name in requested if name in RETM_METRIC_MAP]
    if not active:
        return None

    def collector(output: ModelOutput, inputs: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        if output.retm is None:
            return {}
        target_block = model.retm_target(inputs, target)
        metrics = compute_retm_metrics(
            output.retm,
            target_block,
            output_channels=model.output_channels,
            input_channels=model.input_channels,
        )
        return {name: float(metrics[name].item()) if isinstance(metrics[name], torch.Tensor) else float(metrics[name]) for name in active if name in metrics}

    return collector


def _normalize_profile(value: Optional[str]) -> str:
    return (value or "retm").lower()


def _build_retm_loss_specs(args: argparse.Namespace, model: RTMModel) -> List[LossSpec]:
    def target_transform(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return model.retm_target(inputs, targets)

    return [
        LossSpec(
            "retm_mse",
            MSELoss(),
            "retm",
            weight=args.retm_mse_weight,
            target_transform=target_transform,
        ),
        LossSpec(
            "retm_lstft",
            RetmLSTFTLoss(model.output_channels, model.input_channels),
            "retm",
            weight=args.retm_lstft_weight,
            target_transform=target_transform,
        ),
    ]


def _build_waveform_loss_specs(metrics: Iterable[str], args: argparse.Namespace) -> List[LossSpec]:
    specs: List[LossSpec] = []
    for name in metrics:
        if name == "sdr_db":
            specs.append(LossSpec("sdr_db", SDRLoss(), "waveform", weight=args.time_weight))
        elif name == "mse_db":
            specs.append(LossSpec("mse_db", MSELoss(), "waveform", weight=args.time_weight))
        elif name == "stft_log_ratio":
            specs.append(LossSpec("stft_log_ratio", STFTLogRatioLoss(), "stft", weight=args.stft_weight))
        elif name == "lsd_db":
            specs.append(LossSpec("lsd_db", LSDLoss(), "stft", weight=args.stft_weight))
        elif name == "apd_db":
            specs.append(LossSpec("apd_db", APDLoss(), "stft", weight=args.cs_weight))
        elif name == "complex_cosine":
            specs.append(LossSpec("complex_cosine", ComplexCosineLoss(), "stft", weight=args.cs_weight))
    return specs


def _build_loss_specs(profile: str, args: argparse.Namespace, model: RTMModel) -> List[LossSpec]:
    if profile == "retm":
        return _build_retm_loss_specs(args, model)
    if profile in {"waveforms", "spectral"}:
        specs = _build_waveform_loss_specs(args.metrics.get("waveforms", []), args)
        if not specs:
            raise ValueError("Configure at least one waveform metric for loss_profile='waveforms'.")
        return specs
    if profile == "hybrid":
        retm_specs = _build_retm_loss_specs(args, model)
        waveform_specs = _build_waveform_loss_specs(args.metrics.get("waveforms", []), args)
        if not waveform_specs:
            raise ValueError("Configure at least one waveform metric for loss_profile='hybrid'.")
        return retm_specs + waveform_specs
    raise ValueError(f"Unsupported loss_profile '{args.loss_profile}'. Use 'retm', 'waveforms', or 'hybrid'.")


def _maybe_init_wandb(wandb_cfg: Dict[str, Any], config_payload: Dict[str, Any]):
    if not wandb_cfg.get("enabled"):
        return None
    try:
        import wandb  # type: ignore
    except ImportError:
        print("Weights & Biases logging requested but wandb is not installed. Continuing without it.")
        return None

    run = wandb.init(
        project=wandb_cfg.get("project"),
        entity=wandb_cfg.get("entity"),
        name=wandb_cfg.get("run_name"),
        config=config_payload,
    )
    return run

def main():
    cli_args = parse_args()
    config_path = Path(cli_args.config)
    config = _load_config(config_path)
    merged = _merge_defaults(config)
    required_keys = ["mode", "data_root", "validation_data_root", "test_data_root"]
    missing = [key for key in required_keys if not merged.get(key)]
    if missing:
        raise ValueError(f"Missing required configuration values: {', '.join(missing)}")

    if cli_args.mode:
        merged["mode"] = cli_args.mode
    if cli_args.test:
        merged["test"] = True
    if cli_args.checkpoint is not None:
        merged["checkpoint"] = cli_args.checkpoint

    args = argparse.Namespace(**merged)
    args.config_path = str(config_path)

    save_path = Path(args.save_dir)
    save_path.mkdir(exist_ok=True, parents=True)

    # Load datasets
    dataset = RTMDataset(args.data_root, input_mics=args.input_mics_numbers, target_mics=args.output_mics_numbers)
    val_dataset = RTMDataset(args.validation_data_root, input_mics=args.input_mics_numbers, target_mics=args.output_mics_numbers)
    test_dataset = RTMDataset(args.test_data_root, input_mics=args.input_mics_numbers, target_mics=args.output_mics_numbers)

    if args.mode == "closed_form":
        _run_closed_form(args, dataset, val_dataset, test_dataset, save_path)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    profile = _normalize_profile(args.loss_profile)
    metric_collectors: List[Callable[[ModelOutput, torch.Tensor, torch.Tensor], Dict[str, float]]] = []

    if args.mode == "laenet":
        model = LAENetWrapper(args)
    elif args.mode == "fusnet":
        model = FUSNetWrapper(args)
    elif args.mode == "sconet":
        model = SCONetWrapper(args)
    else:
        raise ValueError(f"Unknown mode '{args.mode}'.")

    loss_specs = _build_loss_specs(profile, args, model)

    waveform_collector = _build_waveform_collector(args.metrics.get("waveforms", []), args.n_fft, args.n_fft // 2)
    if waveform_collector:
        metric_collectors.append(waveform_collector)

    if profile in {"retm", "hybrid"}:
        retm_collector = _build_retm_collector(args.metrics.get("retm", []), model)
        if retm_collector:
            metric_collectors.append(retm_collector)

    wandb_run = _maybe_init_wandb(args.wandb, {k: v for k, v in vars(args).items() if k != "wandb"})

    trainer = Trainer(
        estimator=model,
        loss_specs=loss_specs,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=device,
        grad_clip=args.grad_clip,
        use_amp=args.use_amp,
        metric_collectors=metric_collectors,
        wandb_run=wandb_run,
    )

    if args.test:
        if not args.checkpoint:
            raise ValueError("Provide --checkpoint when running in test mode.")
        print(f"Loading checkpoint {args.checkpoint}")
        state_dict = torch.load(args.checkpoint, map_location=device)
        trainer.estimator.load_state_dict(state_dict)
        print("Evaluating on Test Set...")
        trainer.evaluate(test_dataset, save_dir=save_path, split="test", log=True, record_examples=True)
    else:
        print(f"Training {args.mode} on {device}...")
        trainer.fit(dataset, val_dataset, save_dir=save_path)

    if wandb_run is not None:
        wandb_run.finish()

if __name__ == "__main__":
    main()
