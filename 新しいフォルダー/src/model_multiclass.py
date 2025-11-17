import torch.nn as nn
from .utils import ShibuyaBaseEncoder

class ShibuyaScenarioSoftmax(nn.Module):
    def __init__(self, seq_dim, static_dim, K=5):
        super().__init__()
        self.encoder = ShibuyaBaseEncoder(seq_dim, static_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.encoder.out_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, K)
        )

    def forward(self, seq, static):
        z = self.encoder(seq, static)
        return nn.functional.softmax(self.mlp(z), dim=-1)
