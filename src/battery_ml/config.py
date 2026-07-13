"""Typed configuration loading for reproducible experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class SimulationConfig:
    seed: int = 42
    n_cells: int = 4
    cycles_per_cell: int = 8
    samples_per_cycle: int = 80
    sample_interval_s: float = 10.0
    noise_std: float = 0.008
    temperature_min_c: float = 15.0
    temperature_max_c: float = 35.0
    degradation_rate: float = 0.004
    resistance_growth: float = 0.003
    current_bias_a: float = 0.0
    voltage_bias_v: float = 0.0
    profile: str = "mixed"


@dataclass(frozen=True)
class SplitConfig:
    strategy: str = "held_out_cell"
    validation_fraction: float = 0.2
    test_fraction: float = 0.25
    seed: int = 42


@dataclass(frozen=True)
class ExperimentConfig:
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    output_dir: str = "results"
    window_length: int = 12
    lags: int = 3
    rolling_window: int = 5
    models: tuple[str, ...] = (
        "ridge",
        "random_forest",
        "hist_gradient_boosting",
        "mlp",
        "narx",
        "coulomb_counting",
        "ocv_lookup",
        "ekf",
    )
    seeds: tuple[int, ...] = (42,)
    epochs: int = 16
    batch_size: int = 32
    patience: int = 4
    model_params: dict[str, Any] = field(default_factory=dict)


def load_config(path: str | Path) -> ExperimentConfig:
    """Load a YAML experiment configuration, rejecting unknown nested types naturally."""
    with Path(path).open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    simulation = SimulationConfig(**raw.get("simulation", {}))
    split = SplitConfig(**raw.get("split", {}))
    top_level = {key: value for key, value in raw.items() if key not in {"simulation", "split"}}
    if "models" in top_level:
        top_level["models"] = tuple(top_level["models"])
    if "seeds" in top_level:
        top_level["seeds"] = tuple(top_level["seeds"])
    return ExperimentConfig(simulation=simulation, split=split, **top_level)
