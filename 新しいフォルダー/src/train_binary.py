import torch
from torch.utils.data import DataLoader
from dataset import ShibuyaDataset
from model_binary import ShibuyaBinaryModel
from tqdm import tqdm

def train_binary():
    # dummy sample
    N, T, D = 32, 12, 10
    X_seq = torch.randn(N, T, D)
    X_static = torch.randn(N, 5)
    y = torch.randint(0, 2, (N,1)).float()

    dataset = ShibuyaDataset(X_seq, X_static, y)
    loader = DataLoader(dataset, batch_size=8)

    model = ShibuyaBinaryModel(D, 5)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.BCELoss()

    for epoch in range(5):
        total = 0
        for seq, sta, label in tqdm(loader):
            pred = model(seq, sta)
            loss = loss_fn(pred, label)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"Epoch {epoch+1} loss={total:.4f}")

if __name__ == "__main__":
    train_binary()
