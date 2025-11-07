from __future__ import annotations
import inspect
import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

import torch
from torch.cuda.amp import GradScaler as LegacyGradScaler
from torch.utils.data import DataLoader

from src.models.base import ModelOutput, RTMModel
from src.utils.losses import LossSpec


class Trainer:
    def __init__(
        self,
        estimator: RTMModel,
        loss_specs: Optional[Iterable[LossSpec]] = None,
        loss_fns: Optional[Mapping[str, torch.nn.Module]] = None,
        lr: float = 1e-3,
        epochs: int = 10,
        batch_size: int = 4,
        device: torch.device | str = "cpu",
        grad_clip: Optional[float] = None,
        use_amp: bool = True,
        metric_collectors: Optional[Iterable[Callable[[ModelOutput, torch.Tensor, torch.Tensor], Dict[str, float]]]] = None,
        wandb_run: Optional[object] = None,
    ):
        if loss_specs is None and loss_fns is None:
            raise ValueError("Provide either loss_specs or loss_fns.")

        self.device = torch.device(device)
        self.estimator = estimator.to(self.device)
        self.loss_specs: Sequence[LossSpec] = tuple(loss_specs or ())
        self.loss_fns: Dict[str, torch.nn.Module] = dict(loss_fns or {})
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.grad_clip = grad_clip
        self.use_amp = use_amp
        self.metric_collectors = tuple(metric_collectors or ())
        self.wandb_run = wandb_run
        self._spec_transform_arity: Dict[str, int] = self._infer_transform_arities(self.loss_specs)

        params = list(self.estimator.parameters())
        self.optimizer = torch.optim.AdamW(params, lr=lr) if params else None
        self.scaler = self._init_grad_scaler()

    @staticmethod
    def _infer_transform_arities(specs: Sequence[LossSpec]) -> Dict[str, int]:
        arities: Dict[str, int] = {}
        for spec in specs:
            if spec.target_transform is None:
                arities[spec.name] = 0
                continue
            try:
                signature = inspect.signature(spec.target_transform)
            except (TypeError, ValueError):
                arities[spec.name] = 0
                continue
            count = 0
            for param in signature.parameters.values():
                if param.kind in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                ):
                    count += 1
            arities[spec.name] = count
        return arities

    def _init_grad_scaler(self):
        enabled = self.use_amp and self.optimizer is not None and self.device.type == "cuda"
        amp_scaler_cls = getattr(torch.amp, "GradScaler", None)
        if amp_scaler_cls is not None:
            try:
                return amp_scaler_cls(device_type=self.device.type, enabled=enabled)
            except TypeError:
                return amp_scaler_cls(enabled=enabled)
        return LegacyGradScaler(enabled=enabled)

    def _compute_losses(
        self,
        output: ModelOutput | torch.Tensor,
        inputs: torch.Tensor,
        target_waveform: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if self.loss_specs:
            return self._losses_from_specs(output, inputs, target_waveform)
        return self._losses_from_modules(output, target_waveform)

    def _losses_from_specs(
        self,
        output: ModelOutput | torch.Tensor,
        inputs: torch.Tensor,
        target_waveform: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if not isinstance(output, ModelOutput):
            output = ModelOutput(waveform=output)
        output_dict = output.as_dict()
        target_views: Dict[str, torch.Tensor] = {"waveform": target_waveform}
        losses: Dict[str, torch.Tensor] = {}
        for spec in self.loss_specs:
            pred_value = output_dict.get(spec.prediction_key)
            if pred_value is None:
                continue

            target_value = target_views.get(spec.prediction_key)

            if target_value is None and spec.target_transform is not None:
                arity = self._spec_transform_arity.get(spec.name, 0)
                if arity >= 2:
                    target_value = spec.target_transform(inputs, target_waveform)
                else:
                    target_value = spec.target_transform(target_waveform)
                target_views[spec.prediction_key] = target_value

            if target_value is None:
                target_views.update(self._build_view(spec.prediction_key, inputs, target_waveform))
                target_value = target_views.get(spec.prediction_key)

            if target_value is None:
                raise RuntimeError(f"No target available for prediction key '{spec.prediction_key}'.")

            losses[spec.name] = spec.weight * spec.fn(pred_value, target_value)
        return losses

    def _build_view(
        self,
        key: str,
        inputs: torch.Tensor,
        target_waveform: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if key == "stft":
            stft = self.estimator.stft(target_waveform)
            return {"stft": stft.permute(0, 2, 3, 1)}
        if key == "retm":
            raise RuntimeError("Provide target_transform for ReTM losses.")
        if key == "input":
            return {"input": inputs}
        return {}

    def _losses_from_modules(
        self,
        output: ModelOutput | torch.Tensor,
        target_waveform: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if not self.loss_fns:
            return {}
        tensor_out = output.waveform if isinstance(output, ModelOutput) else output
        return {name: fn(tensor_out, target_waveform) for name, fn in self.loss_fns.items()}

    def _compute_metrics(
        self,
        output: ModelOutput | torch.Tensor,
        inputs: torch.Tensor,
        target_waveform: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if not self.metric_collectors:
            return {}
        if not isinstance(output, ModelOutput):
            output = ModelOutput(waveform=output)
        metrics: Dict[str, torch.Tensor] = {}
        for collector in self.metric_collectors:
            metrics.update(collector(output, inputs, target_waveform))
        return metrics

    def fit(self, train_dataset, val_dataset, save_dir: Path):
        save_dir.mkdir(parents=True, exist_ok=True)
        best_val = float("inf")
        best_path = save_dir / "best_model.pt"
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True)

        for epoch in range(1, self.epochs + 1):
            self.estimator.train()
            running = self._fresh_running_dict()
            for batch in train_loader:
                inputs = batch["input"].to(self.device)
                targets = batch["target"].to(self.device)
                if self.optimizer is not None:
                    self.optimizer.zero_grad(set_to_none=True)
                amp_device = "cuda" if self.device.type == "cuda" else "cpu"
                amp_enabled = self.use_amp and self.optimizer is not None and amp_device == "cuda"
                with torch.amp.autocast(device_type=amp_device, enabled=amp_enabled):
                    output = self.estimator(inputs, targets)
                    losses = self._compute_losses(output, inputs, targets)
                    total = sum(losses.values())
                if self.optimizer is not None and total.requires_grad:
                    self.scaler.scale(total).backward()
                    if self.grad_clip:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.estimator.parameters(), self.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                for name, loss_val in losses.items():
                    running[name] += loss_val.item()
            train_avg = self._average_dict(running, len(train_loader))
            val_losses, val_metrics = self.evaluate(val_loader, save_dir, split="val", log=False)
            val_total = sum(val_losses.values())
            if val_total < best_val:
                best_val = val_total
                if self.optimizer is not None:
                    torch.save(self.estimator.state_dict(), best_path)
            print(f"[Epoch {epoch}] Train {train_avg} | Val {{'losses': {val_losses}, 'metrics': {val_metrics}}}")
            if self.wandb_run is not None:
                wandb_payload = {f"train/{k}": v for k, v in train_avg.items()}
                wandb_payload.update({f"val/loss/{k}": v for k, v in val_losses.items()})
                wandb_payload.update({f"val/metric/{k}": v for k, v in val_metrics.items()})
                wandb_payload["epoch"] = epoch
                self.wandb_run.log(wandb_payload, step=epoch)
        return best_path

    def evaluate(self, dataset_or_loader, save_dir: Path, split: str = "eval", log: bool = True, record_examples: bool = False):
        if hasattr(dataset_or_loader, "__getitem__"):
            loader = DataLoader(dataset_or_loader, batch_size=self.batch_size, shuffle=False, pin_memory=True)
        else:
            loader = dataset_or_loader
        self.estimator.eval()
        totals = self._fresh_running_dict()
        metric_totals: Dict[str, float] = {}
        per_example_records: List[Dict[str, Any]] = []
        example_offset = 0
        with torch.no_grad():
            for batch in loader:
                inputs = batch["input"].to(self.device)
                targets = batch["target"].to(self.device)
                output = self.estimator(inputs, targets)
                losses = self._compute_losses(output, inputs, targets)
                for name, loss_val in losses.items():
                    totals[name] += loss_val.item()
                metrics = self._compute_metrics(output, inputs, targets)
                for name, metric_val in metrics.items():
                    metric_totals[name] = metric_totals.get(name, 0.0) + float(metric_val)
                if record_examples and self.metric_collectors:
                    per_sample = self._collect_per_sample_metrics(output, inputs, targets)
                    paths = batch.get("path")
                    for idx, sample_metrics in enumerate(per_sample):
                        record: Dict[str, Any] = {"index": example_offset + idx}
                        if paths is not None:
                            record["path"] = paths[idx]
                        record.update(sample_metrics)
                        per_example_records.append(record)
                example_offset += inputs.size(0)
        averaged_losses = self._average_dict(totals, len(loader))
        averaged_metrics = self._average_dict(metric_totals, len(loader))
        if record_examples and per_example_records:
            example_metric_avgs = self._average_example_metrics(per_example_records)
            for key, value in example_metric_avgs.items():
                averaged_metrics.setdefault(key, value)
        if log:
            payload = {"losses": averaged_losses, "metrics": averaged_metrics}
            if record_examples:
                payload["examples"] = per_example_records
            (save_dir / f"{split}_results.json").write_text(json.dumps(payload, indent=2))
            print(f"[Eval:{split}] {payload}")
            if self.wandb_run is not None:
                wandb_payload = {f"{split}/loss/{k}": v for k, v in averaged_losses.items()}
                wandb_payload.update({f"{split}/metric/{k}": v for k, v in averaged_metrics.items()})
                self.wandb_run.log(wandb_payload)
        return averaged_losses, averaged_metrics

    def _collect_per_sample_metrics(
        self,
        output: ModelOutput | torch.Tensor,
        inputs: torch.Tensor,
        target_waveform: torch.Tensor,
    ) -> List[Dict[str, float]]:
        if not isinstance(output, ModelOutput):
            output = ModelOutput(waveform=output)
        batch_size = target_waveform.shape[0]
        per_sample: List[Dict[str, float]] = [dict() for _ in range(batch_size)]
        for collector in self.metric_collectors:
            per_sample_fn = getattr(collector, "per_sample", None)
            if per_sample_fn is None:
                continue
            sample_metrics = per_sample_fn(output, inputs, target_waveform)
            for idx, sample_dict in enumerate(sample_metrics):
                if idx < batch_size:
                    per_sample[idx].update(sample_dict)
        return per_sample

    @staticmethod
    def _average_example_metrics(records: List[Dict[str, Any]]) -> Dict[str, float]:
        totals: Dict[str, float] = {}
        counts: Dict[str, int] = {}
        for record in records:
            for key, value in record.items():
                if key in {"index", "path"}:
                    continue
                totals[key] = totals.get(key, 0.0) + float(value)
                counts[key] = counts.get(key, 0) + 1
        return {k: totals[k] / counts[k] for k in totals if counts.get(k)}


    def _fresh_running_dict(self) -> Dict[str, float]:
        if self.loss_specs:
            return {spec.name: 0.0 for spec in self.loss_specs}
        return {name: 0.0 for name in self.loss_fns}

    @staticmethod
    def _average_dict(accumulated: Dict[str, float], steps: int) -> Dict[str, float]:
        if steps == 0:
            return {k: float("nan") for k in accumulated}
        return {k: v / steps for k, v in accumulated.items()}
