import numpy as np
import torch

from battery_ml.models.gradient_boosting import GradientBoostingEstimator
from battery_ml.models.gru import GRUEstimator
from battery_ml.models.linear import RidgeEstimator
from battery_ml.models.mlp import MLPEstimator
from battery_ml.models.narx import NARXEstimator
from battery_ml.models.random_forest import RandomForestEstimator
from battery_ml.models.temporal_cnn import TemporalCNNEstimator, _CausalCNN


def test_classical_estimators_return_finite_predictions() -> None:
    rng = np.random.default_rng(4)
    X = rng.normal(size=(32, 4))
    y = X[:, 0] * 0.5 + X[:, 1] * 0.1
    for model in (
        RidgeEstimator(),
        RandomForestEstimator(n_estimators=8, min_samples_leaf=2),
        GradientBoostingEstimator(),
        MLPEstimator(max_iter=80, alpha=0.001),
    ):
        model.fit(X, y)
        assert np.isfinite(model.predict(X)).all()


def test_narx_recursive_inference_does_not_accept_test_targets() -> None:
    X = np.arange(48, dtype=float).reshape(24, 2) / 24
    y = X[:, 0] * 0.3
    model = NARXEstimator(
        input_delays=2, feedback_delays=2, hidden_layers=(8,), max_iter=80, random_state=7
    )
    model.fit(X, y)
    prediction = model.predict_recursive(X, initial_output=float(y[0]))
    assert prediction.shape == y.shape
    assert np.isfinite(prediction).all()


def test_sequence_estimators_accept_multivariate_windows_and_save_checkpoints(tmp_path) -> None:
    rng = np.random.default_rng(9)
    X = rng.normal(size=(16, 5, 3)).astype(np.float32)
    y = X[:, -1, 0].astype(np.float32)
    for estimator, name in (
        (TemporalCNNEstimator(epochs=2, patience=1), "cnn"),
        (GRUEstimator(epochs=2, patience=1), "gru"),
    ):
        estimator.fit(X[:10], y[:10], X[10:13], y[10:13], tmp_path / f"{name}.pt")
        prediction = estimator.predict(X[13:])
        assert prediction.shape == (3,)
        assert np.isfinite(prediction).all()
        assert (tmp_path / f"{name}.pt").exists()


def test_cnn_convolution_does_not_depend_on_later_window_values() -> None:
    torch.manual_seed(3)
    network = _CausalCNN(features=2, channels=3)
    first = torch.zeros((1, 6, 2))
    second = first.clone()
    second[:, 4:, :] = 100.0
    encoded_one = network.conv(first.transpose(1, 2))[..., :6]
    encoded_two = network.conv(second.transpose(1, 2))[..., :6]
    assert torch.allclose(encoded_one[..., :4], encoded_two[..., :4])


def test_sequence_early_stopping_uses_validation_loss(tmp_path) -> None:
    X = np.ones((12, 4, 2), dtype=np.float32)
    y = np.ones(12, dtype=np.float32)
    model = TemporalCNNEstimator(epochs=8, patience=1, learning_rate=0.0)
    model.fit(X[:8], y[:8], X[8:10], y[8:10], tmp_path / "early-stop.pt")
    assert len(model.training_history) == 2
