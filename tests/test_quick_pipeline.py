import json
from pathlib import Path

from battery_ml.config import ExperimentConfig, SimulationConfig
from battery_ml.experiment import run_full_benchmark


def test_quick_pipeline_writes_soc_soh_and_manifest(tmp_path: Path) -> None:
    config = ExperimentConfig(
        simulation=SimulationConfig(n_cells=4, cycles_per_cell=5, samples_per_cycle=20),
        output_dir=str(tmp_path),
        models=("ridge", "coulomb_counting", "ekf"),
        epochs=3,
    )
    soc, soh = run_full_benchmark(config)
    assert (tmp_path / "soc_metrics.csv").exists()
    assert (tmp_path / "soh_metrics.csv").exists()
    assert (tmp_path / "experiment_manifest.json").exists()
    assert (tmp_path / "soc_true_vs_predicted.png").exists()
    assert set(soc.data_provenance) == {"synthetic data generated inside this project"}
    assert set(soh.data_provenance) == {"synthetic data generated inside this project"}
    manifest = json.loads((tmp_path / "experiment_manifest.json").read_text(encoding="utf-8"))
    assert {"soc_metrics.csv", "soh_metrics.csv"}.issubset(manifest["artifacts"])
