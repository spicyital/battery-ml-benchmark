"""Synthetic distribution-shift robustness experiments."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd

from .config import ExperimentConfig
from .experiment import run_soc_experiment


def run_robustness(config: ExperimentConfig) -> pd.DataFrame:
    """Re-run locked-test SOC evaluation across controlled sensor/environment shifts."""
    scenarios = {
        "nominal": {},
        "increased_sensor_noise": {"noise_std": config.simulation.noise_std * 3},
        "current_sensor_bias": {"current_bias_a": 0.08},
        "voltage_sensor_bias": {"voltage_bias_v": 0.03},
        "unseen_temperature_range": {"temperature_min_c": -5.0, "temperature_max_c": 8.0},
        "unseen_degradation_rate": {"degradation_rate": config.simulation.degradation_rate * 2.5},
        "held_out_cell": {},
        "held_out_load_profile": {"profile": "pulse"},
    }
    records: list[pd.DataFrame] = []
    for scenario, updates in scenarios.items():
        scenario_config = replace(config, simulation=replace(config.simulation, **updates))
        metrics, _ = run_soc_experiment(scenario_config)
        metrics["scenario"] = scenario
        records.append(metrics)
    result = pd.concat(records, ignore_index=True)
    destination = Path(config.output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    result.to_csv(destination / "robustness_metrics.csv", index=False)
    manifest_path = destination / "experiment_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["artifacts"] = sorted(
            set(manifest.get("artifacts", [])) | {"robustness_metrics.csv"}
        )
        manifest["robustness_scenarios"] = list(scenarios)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return result
