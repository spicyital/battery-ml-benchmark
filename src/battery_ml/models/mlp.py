"""Standardized-feature multilayer perceptron estimator."""

from __future__ import annotations

import numpy as np
from sklearn.neural_network import MLPRegressor


class MLPEstimator:
    def __init__(
        self,
        hidden_layer_sizes: tuple[int, ...] = (32,),
        max_iter: int = 300,
        alpha: float = 0.0001,
        random_state: int = 42,
    ) -> None:
        self.model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter,
            alpha=alpha,
            early_stopping=False,
            random_state=random_state,
            learning_rate_init=0.003,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> MLPEstimator:
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
