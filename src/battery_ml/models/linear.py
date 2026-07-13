"""Ridge regression reference ML model."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge


class RidgeEstimator:
    def __init__(self, alpha: float = 1.0, random_state: int = 42) -> None:
        self.model = Ridge(alpha=alpha, random_state=random_state)

    def fit(self, X: np.ndarray, y: np.ndarray) -> RidgeEstimator:
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
