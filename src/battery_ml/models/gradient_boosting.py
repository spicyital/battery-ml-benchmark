"""Histogram Gradient Boosting ML estimator."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor


class GradientBoostingEstimator:
    def __init__(
        self, max_iter: int = 100, learning_rate: float = 0.08, random_state: int = 42
    ) -> None:
        self.model = HistGradientBoostingRegressor(
            max_iter=max_iter, learning_rate=learning_rate, random_state=random_state
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> GradientBoostingEstimator:
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
