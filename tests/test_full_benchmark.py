import json

import numpy as np
import pandas as pd

from battery_ml.config import load_config
from battery_ml.experiment import (
    RELEASE_SOC_MODELS,
    summarize_seed_metrics,
    validate_full_artifacts,
)


def test_full_configuration_covers_release_models_and_protocols() -> None:
    config = load_config("configs/full_benchmark.yaml")
    assert config.full_benchmark
    assert config.seeds == (11, 42, 73)
    assert config.split_strategies == ("held_out_cell", "chronological")
    assert set(RELEASE_SOC_MODELS).issubset(config.models)
    assert {"temporal_cnn", "gru", "narx"}.issubset(config.models)


def test_seed_summary_and_release_schema_validation(tmp_path) -> None:
    rows = pd.DataFrame(
        {
            "task": ["soc", "soc", "soc"],
            "model": ["ridge"] * 3,
            "split_strategy": ["held_out_cell"] * 3,
            "seed": [11, 42, 73],
            "rmse": [0.1, 0.2, 0.3],
            "mae": [0.1, 0.2, 0.3],
            "r2": [0.8, 0.7, 0.6],
            "training_seconds": [1.0, 1.0, 1.0],
            "inference_seconds": [0.1, 0.1, 0.1],
            "runtime_seconds": [1.1, 1.1, 1.1],
            "data_provenance": ["synthetic data generated inside this project"] * 3,
        }
    )
    summary = summarize_seed_metrics(rows)
    assert np.isclose(summary.loc[0, "rmse_mean"], 0.2)
    assert np.isclose(summary.loc[0, "rmse_std"], 0.1)
    directory = tmp_path / "full"
    directory.mkdir()
    rows.to_csv(directory / "soc_metrics_by_seed.csv", index=False)
    rows.assign(task="soh", model="ridge").to_csv(
        directory / "soh_metrics_by_seed.csv", index=False
    )
    summary.to_csv(directory / "soc_metrics_summary.csv", index=False)
    summary.to_csv(directory / "soh_metrics_summary.csv", index=False)
    rows.assign(scenario="nominal").to_csv(directory / "robustness_metrics.csv", index=False)
    rows.to_csv(directory / "runtime_metrics.csv", index=False)
    (directory / "experiment_manifest.json").write_text(
        json.dumps(
            {"data_provenance": "synthetic data generated inside this project", "artifacts": []}
        ),
        encoding="utf-8",
    )
    validate_full_artifacts(directory, required_soc_models={"ridge"}, required_seeds={11, 42, 73})
