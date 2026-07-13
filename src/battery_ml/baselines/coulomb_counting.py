"""Coulomb-counting SOC baseline."""

from __future__ import annotations

import numpy as np
import pandas as pd


def coulomb_counting(
    frame: pd.DataFrame, initial_soc: float = 0.5, nominal_capacity_ah: float = 2.8
) -> np.ndarray:
    """Integrate observed current using declared nominal, non-oracle parameters."""
    prediction = np.empty(len(frame), dtype=float)
    for _, cycle in frame.groupby(["cell_id", "cycle_id"], sort=False):
        ordered = cycle.sort_values("sample_id")
        capacity = nominal_capacity_ah
        soc = initial_soc
        last_time = float(ordered["timestamp_s"].iloc[0])
        for index, row in ordered.iterrows():
            current_time = float(row["timestamp_s"])
            dt = max(0.0, current_time - last_time)
            soc = float(np.clip(soc - float(row["current_a"]) * dt / 3600 / capacity, 0, 1))
            prediction[frame.index.get_loc(index)] = soc
            last_time = current_time
    return prediction
