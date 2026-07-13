"""Experiment orchestration for SOC, SOH, and reproducible manifests."""

from __future__ import annotations

import subprocess
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .baselines.coulomb_counting import coulomb_counting
from .baselines.ekf import ekf_soc
from .baselines.ocv_lookup import ocv_lookup_soc
from .config import ExperimentConfig
from .evaluation import evaluate_predictions, timed_predict
from .features import (
    build_soc_features,
    build_soh_features,
    soc_feature_columns,
    soh_feature_columns,
)
from .models.gradient_boosting import GradientBoostingEstimator
from .models.gru import GRUEstimator
from .models.linear import RidgeEstimator
from .models.mlp import MLPEstimator
from .models.narx import NARXEstimator
from .models.random_forest import RandomForestEstimator
from .models.temporal_cnn import TemporalCNNEstimator
from .preprocessing import TrainOnlyStandardizer
from .reporting import write_reports
from .simulation import simulate_battery_data
from .splits import SplitFrames, make_split
from .windowing import make_windows

BASELINES: dict[str, Callable[[pd.DataFrame], np.ndarray]] = {
    "coulomb_counting": coulomb_counting,
    "ocv_lookup": ocv_lookup_soc,
    "ekf": ekf_soc,
}
SYNTHETIC_PROVENANCE = "synthetic data generated inside this project"


def _model(name: str, params: dict[str, object], seed: int) -> object:
    settings = dict(params.get(name, {}))
    settings["random_state"] = seed
    if name == "ridge":
        return RidgeEstimator(**settings)
    if name == "random_forest":
        return RandomForestEstimator(**settings)
    if name == "hist_gradient_boosting":
        return GradientBoostingEstimator(**settings)
    if name == "mlp":
        if "hidden_layer_sizes" in settings:
            settings["hidden_layer_sizes"] = tuple(settings["hidden_layer_sizes"])
        return MLPEstimator(**settings)
    if name == "narx":
        if "hidden_layers" in settings:
            settings["hidden_layers"] = tuple(settings["hidden_layers"])
        return NARXEstimator(**settings)
    if name == "temporal_cnn":
        return TemporalCNNEstimator(**settings)
    if name == "gru":
        return GRUEstimator(**settings)
    raise ValueError(f"unknown ML model: {name}")


def _split(frame: pd.DataFrame, config: ExperimentConfig) -> SplitFrames:
    return make_split(
        frame,
        config.split.strategy,
        validation_fraction=config.split.validation_fraction,
        test_fraction=config.split.test_fraction,
        seed=config.split.seed,
    )


def _soc_partitions(split: SplitFrames, config: ExperimentConfig) -> tuple[SplitFrames, list[str]]:
    partitions = [
        build_soc_features(part, config.lags, config.rolling_window)
        for part in (split.train, split.validation, split.test)
    ]
    features = soc_feature_columns(partitions[0])
    return SplitFrames(*partitions), features


def run_soc_experiment(
    config: ExperimentConfig, frame: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train/evaluate ML and baseline SOC estimators with a locked test partition."""
    raw = simulate_battery_data(config.simulation) if frame is None else frame
    split, columns = _soc_partitions(_split(raw, config), config)
    scaler = TrainOnlyStandardizer().fit(split.train[columns])
    X_train = scaler.transform(split.train[columns])
    X_test = scaler.transform(split.test[columns])
    y_train = split.train.soc.to_numpy(float)
    y_test = split.test.soc.to_numpy(float)
    records: list[dict[str, float | int | str]] = []
    diagnostics: list[pd.DataFrame] = []
    for seed in config.seeds:
        for name in config.models:
            if name in BASELINES:
                prediction, inference_seconds = timed_predict(BASELINES[name], split.test)
                training_seconds = 0.0
            elif name in {"temporal_cnn", "gru"}:
                sequence_parts = []
                for part in (split.train, split.validation, split.test):
                    scaled = part.copy()
                    scaled.loc[:, columns] = scaler.transform(part[columns])
                    sequence_parts.append(
                        make_windows(scaled, columns, "soc", config.window_length)
                    )
                train_window, validation_window, test_window = sequence_parts
                settings = dict(config.model_params.get(name, {}))
                settings.setdefault("epochs", config.epochs)
                settings.setdefault("batch_size", config.batch_size)
                settings.setdefault("patience", config.patience)
                settings["random_state"] = seed
                model = _model(name, {name: settings}, seed)
                started = time.perf_counter()
                checkpoint = Path(config.output_dir) / f"{name}_seed_{seed}.pt"
                model.fit(  # type: ignore[union-attr]
                    train_window.X,
                    train_window.y,
                    validation_window.X,
                    validation_window.y,
                    checkpoint,
                )
                training_seconds = time.perf_counter() - started
                prediction, inference_seconds = timed_predict(model.predict, test_window.X)  # type: ignore[union-attr]
                y_test = test_window.y
                report_metadata = test_window.metadata.merge(
                    split.test, on=["cell_id", "cycle_id", "sample_id"], how="left"
                )
            else:
                model = _model(name, config.model_params, seed)
                started = time.perf_counter()
                model.fit(X_train, y_train)  # type: ignore[union-attr]
                training_seconds = time.perf_counter() - started
                if name == "narx":
                    prediction, inference_seconds = timed_predict(
                        model.predict_recursive,
                        X_test,
                        0.5,  # type: ignore[union-attr]
                    )
                else:
                    prediction, inference_seconds = timed_predict(model.predict, X_test)  # type: ignore[union-attr]
            prediction = np.clip(prediction, 0, 1)
            if name not in {"temporal_cnn", "gru"}:
                report_metadata = split.test
                y_test = split.test.soc.to_numpy(float)
            metric, diagnostic = evaluate_predictions(
                "soc",
                name,
                y_test,
                prediction,
                report_metadata,
                training_seconds,
                inference_seconds,
                seed,
            )
            metric["data_provenance"] = SYNTHETIC_PROVENANCE
            metric["split_strategy"] = config.split.strategy
            records.append(metric)
            diagnostic["task"] = "soc"
            diagnostic["model"] = name
            diagnostics.append(diagnostic)
    return pd.DataFrame(records), pd.concat(diagnostics, ignore_index=True)


def run_soh_experiment(
    config: ExperimentConfig, frame: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate cycle-level ML estimators from full observed cycle summaries."""
    raw = simulate_battery_data(config.simulation) if frame is None else frame
    raw_split = _split(raw, config)
    split = SplitFrames(
        *(
            build_soh_features(part)
            for part in (raw_split.train, raw_split.validation, raw_split.test)
        )
    )
    columns = soh_feature_columns(split.train)
    scaler = TrainOnlyStandardizer().fit(split.train[columns])
    X_train, X_test = scaler.transform(split.train[columns]), scaler.transform(split.test[columns])
    y_train, y_test = split.train.soh.to_numpy(float), split.test.soh.to_numpy(float)
    records: list[dict[str, float | int | str]] = []
    diagnostics: list[pd.DataFrame] = []
    allowed = {"ridge", "random_forest", "hist_gradient_boosting", "mlp"}
    for seed in config.seeds:
        for name in config.models:
            if name not in allowed:
                continue
            model = _model(name, config.model_params, seed)
            started = time.perf_counter()
            model.fit(X_train, y_train)  # type: ignore[union-attr]
            training_seconds = time.perf_counter() - started
            prediction, inference_seconds = timed_predict(model.predict, X_test)  # type: ignore[union-attr]
            metric, diagnostic = evaluate_predictions(
                "soh",
                name,
                y_test,
                np.clip(prediction, 0, 1),
                split.test,
                training_seconds,
                inference_seconds,
                seed,
            )
            metric["data_provenance"] = SYNTHETIC_PROVENANCE
            metric["split_strategy"] = config.split.strategy
            records.append(metric)
            diagnostic["task"] = "soh"
            diagnostic["model"] = name
            diagnostics.append(diagnostic)
    return pd.DataFrame(records), pd.concat(diagnostics, ignore_index=True)


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-c", "safe.directory=*", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run_full_benchmark(config: ExperimentConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate data once, run SOC/SOH tasks, and write reproducible local artifacts."""
    frame = simulate_battery_data(config.simulation)
    soc, soc_diagnostics = run_soc_experiment(config, frame)
    soh, soh_diagnostics = run_soh_experiment(config, frame)
    manifest = {
        "configuration": asdict(config),
        "git_commit_sha": _git_sha(),
        "data_provenance": SYNTHETIC_PROVENANCE,
        "split_strategy": config.split.strategy,
        "artifacts": ["soc_metrics.csv", "soh_metrics.csv", "prediction_diagnostics.csv"],
    }
    write_reports(
        config.output_dir,
        soc,
        soh,
        pd.concat([soc_diagnostics, soh_diagnostics], ignore_index=True),
        manifest,
    )
    return soc, soh
