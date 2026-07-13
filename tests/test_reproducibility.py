import numpy as np

from battery_ml.models.narx import NARXEstimator


def test_narx_seed_is_deterministic() -> None:
    X = np.arange(40, dtype=float).reshape(20, 2)
    y = X[:, 0] / 30
    one = (
        NARXEstimator(hidden_layers=(6,), max_iter=80, random_state=11)
        .fit(X, y)
        .predict_recursive(X, 0.0)
    )
    two = (
        NARXEstimator(hidden_layers=(6,), max_iter=80, random_state=11)
        .fit(X, y)
        .predict_recursive(X, 0.0)
    )
    np.testing.assert_allclose(one, two)
