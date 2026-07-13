import pandas as pd

from battery_ml.windowing import make_windows


def test_windows_exclude_the_prediction_target_timestep() -> None:
    frame = pd.DataFrame(
        {"cell_id": ["a"] * 5, "cycle_id": [0] * 5, "x": [0, 1, 2, 3, 4], "soc": [0, 1, 2, 3, 4]}
    )
    windows = make_windows(frame, feature_columns=["x"], target_column="soc", window_length=3)
    assert windows.X.shape == (2, 3, 1)
    assert windows.X[0, :, 0].tolist() == [0, 1, 2]
    assert windows.y.tolist() == [3, 4]
    assert windows.metadata.sample_id.tolist() == [3, 4]
