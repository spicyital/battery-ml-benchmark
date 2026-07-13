"""Past-and-present fixed-length sequence construction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WindowedData:
    X: np.ndarray
    y: np.ndarray
    metadata: pd.DataFrame


def make_windows(
    frame: pd.DataFrame, feature_columns: list[str], target_column: str, window_length: int
) -> WindowedData:
    """Predict the next target from a strictly earlier sensor window."""
    if window_length < 1:
        raise ValueError("window_length must be positive")
    values: list[np.ndarray] = []
    targets: list[float] = []
    metadata: list[dict[str, object]] = []
    for (cell_id, cycle_id), group in frame.groupby(["cell_id", "cycle_id"], sort=False):
        group = group.sort_values("sample_id") if "sample_id" in group else group.sort_index()
        matrix = group[feature_columns].to_numpy(dtype=float)
        target = group[target_column].to_numpy(dtype=float)
        for target_index in range(window_length, len(group)):
            values.append(matrix[target_index - window_length : target_index])
            targets.append(target[target_index])
            sample_id = (
                int(group["sample_id"].iloc[target_index])
                if "sample_id" in group
                else int(group.index[target_index])
            )
            metadata.append({"cell_id": cell_id, "cycle_id": cycle_id, "sample_id": sample_id})
    if not values:
        return WindowedData(
            np.empty((0, window_length, len(feature_columns))), np.empty(0), pd.DataFrame(metadata)
        )
    return WindowedData(np.stack(values), np.asarray(targets), pd.DataFrame(metadata))
