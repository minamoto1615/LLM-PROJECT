import torch.nn as nn
from .utils import ShibuyaBaseEncoder

class ShibuyaMultiTask(nn.Module):
    def __init__(self, seq_dim, static_dim):
        super().__init__()
        self.encoder = ShibuyaBaseEncoder(seq_dim, static_dim)

        self.shared = nn.Sequential(
            nn.Linear(self.encoder.out_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        self.head_prob = nn.Sequential(
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        self.head_flow = nn.Linear(64, 1)
        self.head_vacancy = nn.Linear(64, 1)

    def forward(self, seq, static):
        z = self.encoder(seq, static)
        h = self.shared(z)
        return {
            "success": self.head_prob(h),
            "flow": self.head_flow(h),
            "vacancy": self.head_vacancy(h)
        }
