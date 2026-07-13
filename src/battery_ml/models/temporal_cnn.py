"""Compact causal 1D CNN SOC estimator."""

from __future__ import annotations

from torch import Tensor, nn

from .sequence import SequenceRegressor


class _CausalCNN(nn.Module):
    def __init__(self, features: int, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(features, channels, kernel_size=3, padding=2)
        self.head = nn.Sequential(nn.ReLU(), nn.Linear(channels, 1))

    def forward(self, X: Tensor) -> Tensor:
        sequence_length = X.shape[1]
        encoded = self.conv(X.transpose(1, 2))[..., :sequence_length]
        return self.head(encoded[..., -1])


class TemporalCNNEstimator(SequenceRegressor):
    def __init__(self, channels: int = 16, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.channels = channels

    def _build(self, input_features: int) -> nn.Module:
        return _CausalCNN(input_features, self.channels)
