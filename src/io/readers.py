import os
import torchaudio
import torch
from typing import List, Tuple


def list_example_paths(dataset_path: str) -> List[str]:
    return [os.path.join(dataset_path, p) for p in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path,p))]


def load_example_pair(example_path: str, input_mics: List[int], target_mics: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
    recordings = []
    targets = []
    for i in input_mics:
        p = os.path.join(example_path, f"mic_{i}.wav")
        if os.path.isfile(p):
            w, sr = torchaudio.load(p)
            recordings.append(w)
    for i in target_mics:
        p = os.path.join(example_path, f"mic_{i}.wav")
        if os.path.isfile(p):
            w, sr = torchaudio.load(p)
            targets.append(w)
    if not recordings or not targets:
        return None
    inp = torch.stack(recordings, dim=0).squeeze(1)
    tgt = torch.stack(targets, dim=0).squeeze(1)
    min_len = min(inp.shape[1], tgt.shape[1])
    return inp[:, :min_len], tgt[:, :min_len]