import torch
from torch.utils.data import Dataset
from src.io.readers import list_example_paths, load_example_pair

class RTMDataset(Dataset):
    def __init__(self, root_path, input_mics=[4,5,6,7], target_mics=[1,2,3], transform=None):
        self.paths = list_example_paths(root_path)
        self.input_mics = input_mics
        self.target_mics = target_mics
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        data = load_example_pair(p, self.input_mics, self.target_mics)
        if data is None:
            raise IndexError(f"Bad example {p}")
        inp, tgt = data
        if self.transform:
            inp, tgt = self.transform(inp, tgt)
        return {
        "input": inp,
        "target": tgt
        }