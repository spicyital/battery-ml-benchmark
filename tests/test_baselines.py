import numpy as np

from battery_ml.baselines.coulomb_counting import coulomb_counting
from battery_ml.baselines.ekf import ekf_soc
from battery_ml.baselines.ocv_lookup import ocv_lookup_soc
from battery_ml.config import SimulationConfig
from battery_ml.simulation import simulate_battery_data


def test_engineering_baselines_return_finite_soc_predictions() -> None:
    frame = simulate_battery_data(
        SimulationConfig(n_cells=1, cycles_per_cell=2, samples_per_cycle=16)
    )
    for prediction in (coulomb_counting(frame), ocv_lookup_soc(frame), ekf_soc(frame)):
        assert len(prediction) == len(frame)
        assert np.isfinite(prediction).all()
        assert ((prediction >= 0) & (prediction <= 1)).all()


def test_baselines_do_not_require_hidden_true_state_columns() -> None:
    frame = simulate_battery_data(
        SimulationConfig(n_cells=1, cycles_per_cell=2, samples_per_cycle=16)
    )
    observable = frame.drop(columns=["cycle_start_soc", "capacity_ah", "internal_resistance_ohm"])
    for prediction in (
        coulomb_counting(observable),
        ocv_lookup_soc(observable),
        ekf_soc(observable),
    ):
        assert np.isfinite(prediction).all()
