import torch.nn as nn

class ShibuyaBaseEncoder(nn.Module)
    def __init__(self, seq_dim, static_dim, hidden=64)
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=seq_dim,
            hidden_size=hidden,
            batch_first=True
        )
        self.static_dim = static_dim
        self.out_dim = hidden + static_dim

    def forward(self, seq, static)
        _, (h, _) = self.lstm(seq)   # h (1, batch, hidden)
        h_last = h[-1]
        return torch.cat([h_last, static], dim=-1)
