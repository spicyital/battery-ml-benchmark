import numpy as np
import pandas as pd

from battery_ml.evaluation import evaluate_predictions
from battery_ml.metrics import compute_regression_metrics


def test_regression_metrics_are_correct_for_perfect_predictions() -> None:
    metrics = compute_regression_metrics(np.array([0.0, 1.0]), np.array([0.0, 1.0]))
    assert metrics["rmse"] == 0.0
    assert metrics["mae"] == 0.0
    assert metrics["r2"] == 1.0
    assert metrics["max_absolute_error"] == 0.0


def test_soc_slice_metrics_are_finite_when_a_slice_has_no_samples() -> None:
    metadata = pd.DataFrame(
        {"cell_id": ["a", "a"], "cycle_id": [0, 0], "temperature_c": [25.0, 26.0]}
    )
    metrics, _ = evaluate_predictions(
        "soc",
        "ridge",
        np.array([0.4, 0.5]),
        np.array([0.41, 0.49]),
        metadata,
        0.1,
        0.01,
        42,
    )
    assert all(np.isfinite(value) for value in metrics.values() if isinstance(value, float))
    assert metrics["cold_sample_count"] == 0
    assert metrics["soc_0.0_0.2_sample_count"] == 0
