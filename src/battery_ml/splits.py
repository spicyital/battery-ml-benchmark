"""Partition trajectories before preprocessing to prevent train/test leakage."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SplitFrames:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def _copy_partition(frame: pd.DataFrame, values: set[object], column: str) -> pd.DataFrame:
    return (
        frame.loc[frame[column].isin(values)]
        .copy()
        .sort_values(["cell_id", "cycle_id", "sample_id"])
    )


def split_by_cell(
    frame: pd.DataFrame,
    test_fraction: float = 0.25,
    validation_fraction: float = 0.2,
    seed: int = 42,
) -> SplitFrames:
    """Group-split complete cells; no row from an evaluation cell enters training."""
    cells = np.array(sorted(frame["cell_id"].unique()))
    if len(cells) < 3:
        raise ValueError("held-out-cell split needs at least three cells")
    rng = np.random.default_rng(seed)
    rng.shuffle(cells)
    n_test = max(1, int(round(len(cells) * test_fraction)))
    n_validation = max(1, int(round(len(cells) * validation_fraction)))
    if n_test + n_validation >= len(cells):
        n_validation = 1
        n_test = 1
    test = set(cells[:n_test])
    validation = set(cells[n_test : n_test + n_validation])
    train = set(cells[n_test + n_validation :])
    return SplitFrames(
        _copy_partition(frame, train, "cell_id"),
        _copy_partition(frame, validation, "cell_id"),
        _copy_partition(frame, test, "cell_id"),
    )


def split_chronological(
    frame: pd.DataFrame, validation_fraction: float = 0.2, test_fraction: float = 0.2
) -> SplitFrames:
    """Split full cycles in temporal order independently within each cell."""
    sections: list[list[pd.DataFrame]] = [[], [], []]
    for _, cell_frame in frame.groupby("cell_id", sort=False):
        cycles = sorted(cell_frame["cycle_id"].unique())
        if len(cycles) < 3:
            raise ValueError("chronological split needs at least three cycles per cell")
        n_test = max(1, int(round(len(cycles) * test_fraction)))
        n_validation = max(1, int(round(len(cycles) * validation_fraction)))
        n_train = len(cycles) - n_validation - n_test
        if n_train < 1:
            raise ValueError("not enough cycles remaining for training")
        for destination, chosen in zip(
            sections,
            (
                cycles[:n_train],
                cycles[n_train : n_train + n_validation],
                cycles[n_train + n_validation :],
            ),
            strict=False,
        ):
            destination.append(cell_frame.loc[cell_frame["cycle_id"].isin(chosen)])
    return SplitFrames(
        *(
            pd.concat(part, ignore_index=True).sort_values(["cell_id", "cycle_id", "sample_id"])
            for part in sections
        )
    )


def split_held_out_cycles(
    frame: pd.DataFrame, test_cycles: int = 1, validation_cycles: int = 1
) -> SplitFrames:
    """Alias with explicit held-out-cycle counts for experiments."""
    fraction_test = test_cycles / frame["cycle_id"].nunique()
    fraction_validation = validation_cycles / frame["cycle_id"].nunique()
    return split_chronological(frame, fraction_validation, fraction_test)


def make_split(frame: pd.DataFrame, strategy: str, **kwargs: object) -> SplitFrames:
    if strategy in {"held_out_cell", "group_cell", "leave_one_cell_out"}:
        return split_by_cell(
            frame,
            test_fraction=float(kwargs.get("test_fraction", 0.25)),
            validation_fraction=float(kwargs.get("validation_fraction", 0.2)),
            seed=int(kwargs.get("seed", 42)),
        )
    if strategy in {"chronological", "held_out_cycle"}:
        return split_chronological(
            frame,
            validation_fraction=float(kwargs.get("validation_fraction", 0.2)),
            test_fraction=float(kwargs.get("test_fraction", 0.2)),
        )
    raise ValueError(f"unknown split strategy: {strategy}")
