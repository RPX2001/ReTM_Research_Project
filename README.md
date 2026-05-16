ReTM Project

Overview
--------
This repository implements closed-form and neural-network-based ReTM estimators for multi-channel audio denoising/separation.

Quick Start
-----------
**Prerequisites**
- Python 3.10 or newer
- Git, pip
- (Optional) CUDA-enabled GPU and matching PyTorch build for faster training

Create and activate a virtual environment (Windows example):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
```

Install required packages (recommended to use a pinned requirements file if available):

```powershell
pip install torch torchaudio numpy pyyaml tqdm matplotlib
# Optional: wandb
pip install wandb
```

Configuration
-------------
An example YAML config is provided at [example_configs/config_test.yaml](example_configs/config_test.yaml).
The main CLI entrypoint is `main.py`, which reads a config and runs either the closed-form estimator or NN training.

Running
-------
Run training or evaluation using the central CLI:

```powershell
python main.py --config example_configs/config_test.yaml
```

Override the mode, run in test mode, or specify a checkpoint:

```powershell
python main.py --config example_configs/config_test.yaml --mode laenet
python main.py --config example_configs/config_test.yaml --test --checkpoint checkpoints/laenet/best_model.pt
```

Notes:
- Edit `example_configs/config_test.yaml` to point `data_root`, `validation_data_root`, and `test_data_root` to your dataset folders.
- `save_dir` controls where checkpoints and evaluation outputs are written.

Example scripts
---------------
There are example runners in `src/scripts/`:
- [src/scripts/run_closed_form.py](src/scripts/run_closed_form.py) — a simple closed-form evaluation runner (edit paths inside before running).
- [src/scripts/run_nn_example.py](src/scripts/run_nn_example.py) — example training snippet for LAENet (update dataset paths and device). 

Training programmatically
------------------------
If you prefer to run training from a small Python script, an example flow is:

```python
from pathlib import Path
import torch
from src.dataset.dataset import RTMDataset
from src.models.laenet import LAENetWrapper
from src.train.trainer import Trainer

train_ds = RTMDataset('/path/to/train', input_mics=[4,5,6,7], target_mics=[1,2,3])
val_ds = RTMDataset('/path/to/val', input_mics=[4,5,6,7], target_mics=[1,2,3])

model = LAENetWrapper(...)  # consult src/models/laenet.py for constructor args
device = 'cuda' if torch.cuda.is_available() else 'cpu'
trainer = Trainer(estimator=model, loss_fns={'mse': torch.nn.MSELoss()}, lr=1e-3, epochs=100, batch_size=4, device=device)
ckpt = trainer.fit(train_ds, val_ds, save_dir=Path('checkpoints/laenet'))
print('Best checkpoint:', ckpt)
```

Testing / Evaluation
--------------------
To evaluate a trained model, set `test: true` in a config or run with `--test` and provide `--checkpoint`.
Evaluation results and per-split metrics are written to the `save_dir` specified in the config.

Weights & Biases (optional)
---------------------------
W&B logging can be enabled in the YAML config (`wandb.enabled: true`) and configured with `project`, `entity`, and `run_name`.

Data format
-----------
Datasets are loaded via `src/dataset/dataset.py`. Ensure your dataset directory matches the expected structure (mixtures and targets) — inspect `RTMDataset` for details.

Troubleshooting
---------------
- If PyYAML is missing, install it with `pip install pyyaml`.
- Match your PyTorch/CUDA build to your local CUDA drivers when using a GPU.
- Many example scripts contain placeholder paths; update them before running.

Contact
-------
If you want me to add automated setup (requirements.txt) or fix example scripts, tell me which script to update.
