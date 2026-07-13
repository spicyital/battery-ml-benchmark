"""Lightweight GRU SOC estimator for CPU use."""

from __future__ import annotations

from torch import Tensor, nn

from .sequence import SequenceRegressor


class _GRUNetwork(nn.Module):
    def __init__(self, features: int, hidden_size: int) -> None:
        super().__init__()
        self.gru = nn.GRU(features, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, X: Tensor) -> Tensor:
        values, _ = self.gru(X)
        return self.head(values[:, -1])


class GRUEstimator(SequenceRegressor):
    def __init__(self, hidden_size: int = 16, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.hidden_size = hidden_size

    def _build(self, input_features: int) -> nn.Module:
        return _GRUNetwork(input_features, self.hidden_size)
