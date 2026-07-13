"""Deterministic multicell, multicycle synthetic battery trajectories."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import SimulationConfig


def _profile(rng: np.random.Generator, samples: int, kind: str) -> np.ndarray:
    phase = np.linspace(0, np.pi * 2, samples)
    if kind == "urban":
        base = 1.7 * np.sign(np.sin(3 * phase)) + 0.5 * np.sin(9 * phase)
    elif kind == "highway":
        base = 1.2 + 0.25 * np.sin(phase)
    elif kind == "pulse":
        base = np.where((np.arange(samples) // 8) % 2 == 0, 2.2, -1.4)
    else:
        base = 1.6 * np.sin(2 * phase) + 0.55 * np.sin(7 * phase)
    return base + rng.normal(0, 0.15, samples)


def simulate_battery_data(config: SimulationConfig) -> pd.DataFrame:
    """Generate realistic enough, non-proprietary benchmark data for ML experiments."""
    rng = np.random.default_rng(config.seed)
    rows: list[dict[str, float | int | str]] = []
    profiles = ("urban", "highway", "pulse") if config.profile == "mixed" else (config.profile,)
    for cell_index in range(config.n_cells):
        cell_id = f"cell_{cell_index:02d}"
        nominal_capacity = 2.8 * (1 + rng.normal(0, 0.025))
        base_resistance = 0.045 * (1 + rng.normal(0, 0.08))
        soc = float(rng.uniform(0.45, 0.85))
        for cycle_id in range(config.cycles_per_cell):
            soh = max(0.55, 1 - config.degradation_rate * cycle_id * (1 + 0.15 * cell_index))
            capacity_ah = nominal_capacity * soh
            resistance = base_resistance * (
                1 + config.resistance_growth * cycle_id * (1 + cell_index)
            )
            profile_name = profiles[(cell_index + cycle_id) % len(profiles)]
            current = _profile(rng, config.samples_per_cycle, profile_name)
            direction = 1 if cycle_id % 2 else -1
            current = np.abs(current) * direction
            initial_soc = soc
            for sample_index, ideal_current in enumerate(current):
                elapsed = sample_index * config.sample_interval_s
                temperature = (
                    (config.temperature_min_c + config.temperature_max_c) / 2
                    + (config.temperature_max_c - config.temperature_min_c)
                    / 2
                    * np.sin((cycle_id + sample_index / config.samples_per_cycle) * 0.8)
                    + cell_index * 0.35
                )
                temperature += rng.normal(0, 0.35)
                effective_capacity = capacity_ah * (1 - 0.0015 * max(0, 25 - temperature))
                soc = float(
                    np.clip(
                        soc - ideal_current * config.sample_interval_s / 3600 / effective_capacity,
                        0.02,
                        0.98,
                    )
                )
                ocv = 3.0 + 1.15 * soc - 0.10 * soc * (1 - soc)
                terminal_voltage = ocv - ideal_current * resistance + 0.002 * (temperature - 25)
                measured_current = (
                    ideal_current + config.current_bias_a + rng.normal(0, config.noise_std)
                )
                measured_voltage = (
                    terminal_voltage + config.voltage_bias_v + rng.normal(0, config.noise_std)
                )
                segment = "charge" if ideal_current < 0 else "discharge"
                rows.append(
                    {
                        "timestamp_s": (cycle_id * config.samples_per_cycle + sample_index)
                        * config.sample_interval_s,
                        "cell_id": cell_id,
                        "cycle_id": cycle_id,
                        "sample_id": sample_index,
                        "profile": profile_name,
                        "segment": segment,
                        "current_a": measured_current,
                        "voltage_v": measured_voltage,
                        "temperature_c": temperature,
                        "soc": soc,
                        "capacity_ah": capacity_ah,
                        "soh": soh,
                        "internal_resistance_ohm": resistance,
                        "elapsed_cycle_s": elapsed,
                        "cycle_start_soc": initial_soc,
                    }
                )
    return pd.DataFrame(rows)
