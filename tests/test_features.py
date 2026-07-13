import pandas as pd

from battery_ml.features import build_soc_features


def test_lag_and_rolling_features_are_past_only() -> None:
    frame = pd.DataFrame(
        {
            "cell_id": ["a"] * 4,
            "cycle_id": [0] * 4,
            "current_a": [1.0, 2.0, 3.0, 4.0],
            "voltage_v": [3.0, 3.1, 3.2, 3.3],
            "temperature_c": [25.0] * 4,
            "elapsed_cycle_s": [0, 1, 2, 3],
            "soc": [0.9, 0.8, 0.7, 0.6],
        }
    )
    features = build_soc_features(frame, lags=1, rolling_window=2)
    assert features.loc[2, "current_a_lag_1"] == 2.0
    assert features.loc[2, "current_a_roll_mean"] == 2.5
    assert features.loc[0, "current_a_lag_1"] != features.loc[0, "current_a"]
