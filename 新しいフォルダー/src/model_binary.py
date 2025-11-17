import torch.nn as nn
from .utils import ShibuyaBaseEncoder

class ShibuyaBinaryModel(nn.Module):
    def __init__(self, seq_dim, static_dim):
        super().__init__()
        self.encoder = ShibuyaBaseEncoder(seq_dim, static_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.encoder.out_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, seq, static):
        z = self.encoder(seq, static)
        return self.sigmoid(self.mlp(z))
