"""Evaluation records and diagnostic error slices."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .metrics import compute_regression_metrics, final_cycle_drift


def evaluate_predictions(
    task: str,
    model: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metadata: pd.DataFrame,
    train_seconds: float,
    inference_seconds: float,
    seed: int,
) -> tuple[dict[str, float | int | str], pd.DataFrame]:
    """Return summary metrics plus row-level diagnostics for reporting."""
    metrics: dict[str, float | int | str] = compute_regression_metrics(y_true, y_pred)
    metrics.update(
        {
            "task": task,
            "model": model,
            "seed": seed,
            "training_seconds": train_seconds,
            "inference_seconds": inference_seconds,
            "runtime_seconds": train_seconds + inference_seconds,
        }
    )
    diagnostic = metadata.reset_index(drop=True).copy()
    diagnostic["y_true"] = np.asarray(y_true)
    diagnostic["y_pred"] = np.asarray(y_pred)
    diagnostic["absolute_error"] = np.abs(diagnostic["y_pred"] - diagnostic["y_true"])
    if task == "soc":
        metrics["final_cycle_drift"] = final_cycle_drift(y_true, y_pred)
        if "temperature_c" in diagnostic:
            for name, mask in {
                "cold": diagnostic.temperature_c < 20,
                "hot": diagnostic.temperature_c > 30,
            }.items():
                count = int(mask.sum())
                metrics[f"{name}_sample_count"] = count
                metrics[f"{name}_mae"] = (
                    float(diagnostic.loc[mask, "absolute_error"].mean()) if count else 0.0
                )
        for low, high in ((0.0, 0.2), (0.2, 0.8), (0.8, 1.01)):
            mask = diagnostic.y_true.between(low, high, inclusive="left")
            prefix = f"soc_{low:.1f}_{high:.1f}"
            count = int(mask.sum())
            metrics[f"{prefix}_sample_count"] = count
            metrics[f"{prefix}_mae"] = (
                float(diagnostic.loc[mask, "absolute_error"].mean()) if count else 0.0
            )
    return metrics, diagnostic


def timed_predict(function: object, *args: object) -> tuple[np.ndarray, float]:
    started = time.perf_counter()
    prediction = function(*args)  # type: ignore[operator]
    return np.asarray(prediction, dtype=float), time.perf_counter() - started
