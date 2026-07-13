"""Simple synthetic OCV inverse-map baseline."""

from __future__ import annotations

import numpy as np
import pandas as pd


def ocv_lookup_soc(frame: pd.DataFrame) -> np.ndarray:
    """Invert the simulator's monotonic OCV approximation, intentionally ignoring dynamics."""
    compensated_voltage = (
        frame["voltage_v"].to_numpy(float) + frame["current_a"].to_numpy(float) * 0.045
    )
    return np.clip((compensated_voltage - 3.0) / 1.15, 0, 1)
