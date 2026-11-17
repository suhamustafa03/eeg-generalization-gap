"""
Train and checkpoint the CNN baseline on a given processed dataset.

Usage:
    python src/train_cnn.py --data data/processed/cassette.npz \
        --out results/cnn_cassette.pt --epochs 15
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from models import CNNBaseline, normalize_epochs


def train(data_path: pathlib.Path, out_path: pathlib.Path, n_epochs: int, seed: int = 0):
    torch.manual_seed(seed)
    np.random.seed(seed)

    npz = np.load(data_path, allow_pickle=True)
    epochs = normalize_epochs(npz["epochs"])
    labels = npz["labels"]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    X = torch.tensor(epochs, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)

    # Simple 80/20 subject-naive split for training-time validation only;
    # the real evaluation (within vs. cross-cohort) happens in evaluate.py
    # using subject-level splits so there is no leakage there.
    n = len(X)
    perm = np.random.permutation(n)
    split = int(0.8 * n)
    train_idx, val_idx = perm[:split], perm[split:]

    train_ds = TensorDataset(X[train_idx], y[train_idx])
    val_ds = TensorDataset(X[val_idx], y[val_idx])
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256)

    model = CNNBaseline().to(device)

    class_counts = np.bincount(labels, minlength=5)
    class_weights = torch.tensor(
        (class_counts.sum() / (class_counts + 1e-6)), dtype=torch.float32
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb).argmax(dim=1)
                correct += (pred == yb).sum().item()
                total += len(yb)
        print(
            f"epoch {epoch + 1}/{n_epochs}  "
            f"train_loss={total_loss / len(train_ds):.4f}  val_acc={correct / total:.4f}"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)
    print(f"Saved model to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--epochs", type=int, default=15)
    args = parser.parse_args()
    train(args.data, args.out, args.epochs)
