"""One-state equivalent-circuit Extended Kalman Filter baseline."""

from __future__ import annotations

import numpy as np
import pandas as pd


def ekf_soc(
    frame: pd.DataFrame,
    process_variance: float = 1e-5,
    measurement_variance: float = 3e-4,
    initial_soc: float = 0.5,
    nominal_capacity_ah: float = 2.8,
    resistance_ohm: float = 0.045,
) -> np.ndarray:
    """Estimate SOC from observed current/voltage with declared non-oracle parameters."""
    prediction = np.empty(len(frame), dtype=float)
    for _, cycle in frame.groupby(["cell_id", "cycle_id"], sort=False):
        ordered = cycle.sort_values("sample_id")
        state = initial_soc
        covariance = 0.02
        capacity = nominal_capacity_ah
        last_time = float(ordered["timestamp_s"].iloc[0])
        for index, row in ordered.iterrows():
            now = float(row["timestamp_s"])
            dt = max(0.0, now - last_time)
            state = float(np.clip(state - float(row["current_a"]) * dt / 3600 / capacity, 0, 1))
            covariance += process_variance
            expected_voltage = 3.0 + 1.15 * state - float(row["current_a"]) * resistance_ohm
            innovation = float(row["voltage_v"]) - expected_voltage
            h = 1.15
            gain = covariance * h / (h * covariance * h + measurement_variance)
            state = float(np.clip(state + gain * innovation, 0, 1))
            covariance = (1 - gain * h) * covariance
            prediction[frame.index.get_loc(index)] = state
            last_time = now
    return prediction
