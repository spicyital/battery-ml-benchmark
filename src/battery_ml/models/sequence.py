"""Shared CPU-only PyTorch dataset and early-stopping trainer."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset


class WindowDataset(Dataset[tuple[Tensor, Tensor]]):
    """Multivariate time-series windows with scalar regression targets."""

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return self.X[index], self.y[index]


class SequenceRegressor:
    """Shared fit/predict wrapper; concrete classes supply a causal network."""

    def __init__(
        self,
        epochs: int = 16,
        batch_size: int = 32,
        patience: int = 4,
        random_state: int = 42,
        learning_rate: float = 0.003,
    ) -> None:
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.random_state = random_state
        self.learning_rate = learning_rate
        self.network: nn.Module | None = None
        self.training_history: list[dict[str, float]] = []

    def _build(self, input_features: int) -> nn.Module:
        raise NotImplementedError

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_validation: np.ndarray,
        y_validation: np.ndarray,
        checkpoint_path: str | Path,
    ) -> SequenceRegressor:
        torch.manual_seed(self.random_state)
        torch.use_deterministic_algorithms(True, warn_only=True)
        self.network = self._build(int(X_train.shape[2])).cpu()
        train_loader = DataLoader(
            WindowDataset(X_train, y_train),
            batch_size=self.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.random_state),
        )
        validation_loader = DataLoader(
            WindowDataset(X_validation, y_validation), batch_size=self.batch_size, shuffle=False
        )
        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate)
        loss_fn = nn.MSELoss()
        destination = Path(checkpoint_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        best_loss, wait = float("inf"), 0
        for epoch in range(1, self.epochs + 1):
            self.network.train()
            train_losses: list[float] = []
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                loss = loss_fn(self.network(X_batch).squeeze(-1), y_batch)
                loss.backward()
                optimizer.step()
                train_losses.append(float(loss.detach()))
            self.network.eval()
            validation_losses: list[float] = []
            with torch.no_grad():
                for X_batch, y_batch in validation_loader:
                    validation_losses.append(
                        float(loss_fn(self.network(X_batch).squeeze(-1), y_batch))
                    )
            train_loss = float(np.mean(train_losses))
            validation_loss = float(np.mean(validation_losses))
            self.training_history.append(
                {
                    "epoch": float(epoch),
                    "training_loss": train_loss,
                    "validation_loss": validation_loss,
                }
            )
            if validation_loss < best_loss - 1e-8:
                best_loss, wait = validation_loss, 0
                torch.save(
                    {"state_dict": self.network.state_dict(), "history": self.training_history},
                    destination,
                )
            else:
                wait += 1
                if wait >= self.patience:
                    break
        payload = torch.load(destination, map_location="cpu", weights_only=True)
        self.network.load_state_dict(payload["state_dict"])
        destination.with_suffix(".history.json").write_text(
            json.dumps(self.training_history, indent=2), encoding="utf-8"
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.network is None:
            raise RuntimeError("sequence model must be fit before prediction")
        self.network.eval()
        with torch.no_grad():
            return self.network(torch.as_tensor(X, dtype=torch.float32)).squeeze(-1).numpy()
