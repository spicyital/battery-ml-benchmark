"""Regression metrics and battery-specific error slices."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return finite standard metrics for a one-dimensional regression task."""
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    error = predicted - actual
    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "mae": float(mean_absolute_error(actual, predicted)),
        "r2": float(r2_score(actual, predicted)) if len(actual) > 1 else float("nan"),
        "max_absolute_error": float(np.max(np.abs(error))) if len(error) else float("nan"),
        "mean_bias": float(np.mean(error)) if len(error) else float("nan"),
    }
    nonzero = np.abs(actual) > 1e-9
    if nonzero.any():
        metrics["mape"] = float(np.mean(np.abs(error[nonzero] / actual[nonzero])) * 100)
    return metrics


def final_cycle_drift(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.asarray(y_pred)[-1] - np.asarray(y_true)[-1])
