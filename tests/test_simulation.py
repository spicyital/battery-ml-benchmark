from dataclasses import asdict

import pandas as pd

from battery_ml.config import SimulationConfig
from battery_ml.simulation import simulate_battery_data


def test_simulation_is_deterministic_and_has_valid_ranges() -> None:
    config = SimulationConfig(seed=7, n_cells=2, cycles_per_cell=3, samples_per_cycle=12)
    left = simulate_battery_data(config)
    right = simulate_battery_data(config)

    pd.testing.assert_frame_equal(left, right)
    assert set(asdict(config)).issubset(
        {
            "seed",
            "n_cells",
            "cycles_per_cell",
            "samples_per_cycle",
            "sample_interval_s",
            "noise_std",
            "temperature_min_c",
            "temperature_max_c",
            "degradation_rate",
            "resistance_growth",
            "current_bias_a",
            "voltage_bias_v",
            "profile",
        }
    )
    assert left["soc"].between(0, 1).all()
    assert left["soh"].between(0.5, 1).all()
    assert {"charge", "discharge"}.issubset(set(left["segment"]))
