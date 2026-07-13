"""Train-only standardization helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class TrainOnlyStandardizer:
    """A deliberately small wrapper exposing the training fit statistics for tests/audits."""

    def __init__(self) -> None:
        self._scaler = StandardScaler()
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.columns_: list[str] | None = None

    def fit(self, train: pd.DataFrame | np.ndarray) -> TrainOnlyStandardizer:
        values = (
            train.to_numpy(dtype=float)
            if isinstance(train, pd.DataFrame)
            else np.asarray(train, dtype=float)
        )
        self._scaler.fit(values)
        self.mean_ = self._scaler.mean_.copy()
        self.scale_ = self._scaler.scale_.copy()
        self.columns_ = list(train.columns) if isinstance(train, pd.DataFrame) else None
        return self

    def transform(self, frame: pd.DataFrame | np.ndarray) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("standardizer must be fit on training data before transform")
        values = (
            frame.to_numpy(dtype=float)
            if isinstance(frame, pd.DataFrame)
            else np.asarray(frame, dtype=float)
        )
        return self._scaler.transform(values)
