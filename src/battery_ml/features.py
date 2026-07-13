"""Causal feature engineering for time-series and cycle-level ML tasks."""

from __future__ import annotations

import numpy as np
import pandas as pd

RAW_SOC_FEATURES = ["current_a", "voltage_v", "temperature_c", "elapsed_cycle_s"]


def build_soc_features(frame: pd.DataFrame, lags: int = 3, rolling_window: int = 5) -> pd.DataFrame:
    """Build per-row features using only a row and observations preceding it."""
    sort_columns = [column for column in ["cell_id", "cycle_id", "sample_id"] if column in frame]
    result = frame.copy().sort_values(sort_columns).reset_index(drop=True)
    groups = result.groupby(["cell_id", "cycle_id"], sort=False)
    for column in ("current_a", "voltage_v", "temperature_c"):
        for lag in range(1, lags + 1):
            result[f"{column}_lag_{lag}"] = groups[column].shift(lag)
        result[f"{column}_roll_mean"] = groups[column].transform(
            lambda values: values.rolling(rolling_window, min_periods=1).mean()
        )
        result[f"{column}_roll_std"] = groups[column].transform(
            lambda values: values.rolling(rolling_window, min_periods=1).std().fillna(0)
        )
        result[f"{column}_derivative"] = groups[column].diff().fillna(0)
    elapsed_delta = groups["elapsed_cycle_s"].diff().fillna(0).clip(lower=0)
    result["cumulative_ah"] = (result["current_a"] * elapsed_delta).groupby(
        [result["cell_id"], result["cycle_id"]], sort=False
    ).cumsum() / 3600
    result["is_charging"] = (result["current_a"] < 0).astype(int)
    result["temperature_current"] = result["temperature_c"] * result["current_a"]
    # Missing lags are deliberately filled from the first observable value, never a future value.
    for column in result.columns:
        if "_lag_" in column:
            result[column] = result.groupby(["cell_id", "cycle_id"], sort=False)[column].transform(
                lambda values: values.ffill().fillna(0)
            )
    return result.replace([np.inf, -np.inf], 0).fillna(0)


def soc_feature_columns(frame: pd.DataFrame) -> list[str]:
    """Return observable sensor features and causal derivatives only.

    Latent simulator fields (true SOC/SOH, capacity, resistance, and cycle-start SOC) are
    deliberately excluded so they cannot leak labels into ML training.
    """
    explicit = {"cumulative_ah", "is_charging", "temperature_current"}
    prefixes = ("current_a", "voltage_v", "temperature_c")
    return [
        column
        for column in frame.columns
        if pd.api.types.is_numeric_dtype(frame[column])
        and (column in RAW_SOC_FEATURES or column in explicit or column.startswith(prefixes))
    ]


def build_soh_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate a complete observed cycle into cycle-level SOH predictors."""
    rows: list[dict[str, float | int | str]] = []
    for (cell_id, cycle_id), cycle in frame.groupby(["cell_id", "cycle_id"], sort=False):
        discharge = cycle.loc[cycle["current_a"] > 0]
        charge = cycle.loc[cycle["current_a"] < 0]
        voltage_range = float(cycle["voltage_v"].max() - cycle["voltage_v"].min())
        row: dict[str, float | int | str] = {
            "cell_id": cell_id,
            "cycle_id": cycle_id,
            "estimated_discharge_capacity_ah": float(discharge["current_a"].sum() * 10 / 3600),
            "voltage_mean": float(cycle["voltage_v"].mean()),
            "voltage_std": float(cycle["voltage_v"].std(ddof=0)),
            "voltage_range": voltage_range,
            "current_mean": float(cycle["current_a"].mean()),
            "current_std": float(cycle["current_a"].std(ddof=0)),
            "temperature_mean": float(cycle["temperature_c"].mean()),
            "temperature_std": float(cycle["temperature_c"].std(ddof=0)),
            "resistance_proxy": float(voltage_range / (cycle["current_a"].abs().mean() + 1e-6)),
            "charge_duration_s": float(len(charge) * 10),
            "discharge_duration_s": float(len(discharge) * 10),
            "historical_cycle": int(cycle_id),
            "soh": float(cycle["soh"].iloc[-1]),
        }
        rows.append(row)
    result = pd.DataFrame(rows)
    result["capacity_fade_proxy"] = result.groupby("cell_id")[
        "estimated_discharge_capacity_ah"
    ].transform(lambda values: values / max(float(values.iloc[0]), 1e-6))
    return result


def soh_feature_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column not in {"cell_id", "cycle_id", "soh"}]
