import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

def create_sequences(data: pd.DataFrame, window_size: int):
    features = data.drop(columns=['target']).to_numpy()
    targets = data['target'].to_numpy()

    xs = []
    ys = []

    for i in range(len(data) - window_size + 1):
        x = features[i : i + window_size]
        y = targets[i + window_size - 1] # take the last target in the current window
        xs.append(x)
        ys.append(y)

    xs = np.array(xs)
    ys = np.array(ys)
    return xs, ys


class VolatilityDataset(Dataset):
    def __init__(self, data, window_size):
        super().__init__()

        xs, ys = create_sequences(data, window_size)

        self.xs = torch.tensor(xs, dtype=torch.float32)
        self.ys = torch.tensor(ys, dtype=torch.float32)

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        return (self.xs[idx], self.ys[idx])