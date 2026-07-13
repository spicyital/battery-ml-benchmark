"""Configurable NARX-style state estimator with safe recursive inference."""

from __future__ import annotations

import numpy as np
from sklearn.neural_network import MLPRegressor


class NARXEstimator:
    """Uses lagged inputs plus lagged outputs in training, then its own predictions at inference."""

    def __init__(
        self,
        input_delays: int = 3,
        feedback_delays: int = 2,
        hidden_layers: tuple[int, ...] = (24,),
        max_iter: int = 250,
        random_state: int = 42,
        training_restarts: int = 1,
    ) -> None:
        self.input_delays = input_delays
        self.feedback_delays = feedback_delays
        self.hidden_layers = hidden_layers
        self.max_iter = max_iter
        self.random_state = random_state
        self.training_restarts = training_restarts
        self.model: MLPRegressor | None = None

    def _design(self, X: np.ndarray, feedback: np.ndarray) -> np.ndarray:
        rows: list[np.ndarray] = []
        for index in range(len(X)):
            inputs = [X[max(0, index - delay)] for delay in range(self.input_delays + 1)]
            outputs = [
                feedback[max(0, index - delay)] for delay in range(1, self.feedback_delays + 1)
            ]
            rows.append(np.concatenate([*inputs, np.asarray(outputs)]))
        return np.asarray(rows)

    def fit(self, X: np.ndarray, y: np.ndarray) -> NARXEstimator:
        features = self._design(np.asarray(X, dtype=float), np.asarray(y, dtype=float))
        best: MLPRegressor | None = None
        best_loss = float("inf")
        for restart in range(self.training_restarts):
            candidate = MLPRegressor(
                hidden_layer_sizes=self.hidden_layers,
                max_iter=self.max_iter,
                random_state=self.random_state + restart,
                early_stopping=False,
                solver="lbfgs",
            ).fit(features, y)
            loss = float(candidate.loss_)
            if loss < best_loss:
                best, best_loss = candidate, loss
        self.model = best
        return self

    def predict_recursive(self, X: np.ndarray, initial_output: float = 0.5) -> np.ndarray:
        """Closed-loop prediction: only previous predictions become feedback features."""
        if self.model is None:
            raise RuntimeError("NARXEstimator must be fit before prediction")
        inputs = np.asarray(X, dtype=float)
        outputs = np.full(len(inputs), float(initial_output), dtype=float)
        for index in range(len(inputs)):
            design = self._design(inputs[: index + 1], outputs[: index + 1])[-1:]
            outputs[index] = float(self.model.predict(design)[0])
        return outputs

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_recursive(X)
