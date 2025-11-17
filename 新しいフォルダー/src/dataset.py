import torch
from torch.utils.data import Dataset

class ShibuyaDataset(Dataset):
    def __init__(self, X_seq, X_static, y):
        self.X_seq = X_seq
        self.X_static = X_static
        self.y = y

    def __len__(self):
        return self.X_seq.shape[0]

    def __getitem__(self, idx):
        return (
            self.X_seq[idx],
            self.X_static[idx],
            self.y[idx]
        )
