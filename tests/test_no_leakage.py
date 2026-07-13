import numpy as np
import pandas as pd

from battery_ml.config import SimulationConfig
from battery_ml.features import build_soh_features, soc_feature_columns, soh_feature_columns
from battery_ml.models.mlp import MLPEstimator
from battery_ml.preprocessing import TrainOnlyStandardizer
from battery_ml.simulation import simulate_battery_data


def test_standardizer_fits_training_rows_only() -> None:
    train = pd.DataFrame({"x": [0.0, 2.0]})
    test = pd.DataFrame({"x": [100.0]})
    scaler = TrainOnlyStandardizer().fit(train)
    transformed = scaler.transform(test)
    assert scaler.mean_[0] == 1.0
    assert np.isclose(transformed[0, 0], 99.0)


def test_soc_model_features_exclude_latent_state_and_target_columns() -> None:
    frame = pd.DataFrame(
        {
            "current_a": [1.0],
            "voltage_v": [3.7],
            "temperature_c": [25.0],
            "elapsed_cycle_s": [0.0],
            "cycle_start_soc": [0.8],
            "capacity_ah": [2.7],
            "internal_resistance_ohm": [0.05],
            "soc": [0.8],
            "soh": [0.99],
            "cell_id": ["cell_00"],
            "cycle_id": [0],
            "sample_id": [0],
            "profile": ["urban"],
            "segment": ["discharge"],
        }
    )
    columns = soc_feature_columns(frame)
    assert {"cycle_start_soc", "capacity_ah", "internal_resistance_ohm", "soc", "soh"}.isdisjoint(
        columns
    )
    assert {"current_a", "voltage_v", "temperature_c", "elapsed_cycle_s"}.issubset(columns)


def test_mlp_does_not_make_a_random_internal_validation_split() -> None:
    assert MLPEstimator().model.early_stopping is False


def test_soh_features_exclude_simulator_targets_and_latent_parameters() -> None:
    frame = simulate_battery_data(
        SimulationConfig(n_cells=1, cycles_per_cell=3, samples_per_cycle=12)
    )
    features = build_soh_features(frame)
    columns = soh_feature_columns(features)
    assert {"soh", "capacity_ah", "internal_resistance_ohm", "cycle_start_soc"}.isdisjoint(columns)
