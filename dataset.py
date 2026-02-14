import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

def create_sequences(features: np.ndarray, targets: np.ndarray, window_size: int):
    if len(features) != len(targets):
        raise ValueError(f'Невідповідність розмірів вхідних масивів')

    xs = []
    ys = []

    for i in range(len(features) - window_size + 1):
        x = features[i : i + window_size]
        y = targets[i + window_size - 1] # take the last target in the current window
        xs.append(x)
        ys.append(y)

    xs = np.array(xs)
    ys = np.array(ys)
    return xs, ys


class VolatilityDataset(Dataset):
    def __init__(self, features: np.ndarray, targets: np.ndarray, window_size: int):
        super().__init__()

        xs, ys = create_sequences(features, targets, window_size)

        self.xs = torch.tensor(xs, dtype=torch.float32)
        self.ys = torch.tensor(ys, dtype=torch.float32)

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        return (self.xs[idx], self.ys[idx])