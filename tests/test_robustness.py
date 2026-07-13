from pathlib import Path

from battery_ml.config import ExperimentConfig, SimulationConfig
from battery_ml.robustness import run_robustness


def test_robustness_writes_named_scenarios(tmp_path: Path) -> None:
    config = ExperimentConfig(
        simulation=SimulationConfig(n_cells=4, cycles_per_cell=5, samples_per_cycle=16),
        output_dir=str(tmp_path),
        models=("ridge", "coulomb_counting"),
    )
    results = run_robustness(config)
    assert {"increased_sensor_noise", "current_sensor_bias", "held_out_cell"}.issubset(
        set(results.scenario)
    )
    assert (tmp_path / "robustness_metrics.csv").exists()
