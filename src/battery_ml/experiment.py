"""Experiment orchestration for SOC, SOH, and reproducible manifests."""

from __future__ import annotations

import json
import subprocess
import time
from collections.abc import Callable
from dataclasses import asdict, replace
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
from .reporting import write_full_release_artifacts, write_reports
from .simulation import simulate_battery_data
from .splits import SplitFrames, make_split
from .windowing import make_windows

BASELINES: dict[str, Callable[[pd.DataFrame], np.ndarray]] = {
    "coulomb_counting": coulomb_counting,
    "ocv_lookup": ocv_lookup_soc,
    "ekf": ekf_soc,
}
SYNTHETIC_PROVENANCE = "synthetic data generated inside this project"
RELEASE_SOC_MODELS = (
    "ridge",
    "random_forest",
    "hist_gradient_boosting",
    "mlp",
    "narx",
    "temporal_cnn",
    "gru",
    "coulomb_counting",
    "ocv_lookup",
    "ekf",
)


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
                    scaled = scaled.astype({column: float for column in columns})
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
                checkpoint = (
                    Path(config.output_dir) / f"{name}_{config.split.strategy}_seed_{seed}.pt"
                )
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
            metric["hyperparameters"] = json.dumps(
                {} if name in BASELINES else config.model_params.get(name, {}), sort_keys=True
            )
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
            metric["hyperparameters"] = json.dumps(
                config.model_params.get(name, {}), sort_keys=True
            )
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


def summarize_seed_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    """Aggregate locked-test metrics without hiding the per-seed source rows."""
    if metrics.empty:
        return pd.DataFrame()
    group_columns = ["task", "model", "split_strategy"]
    numeric = metrics.select_dtypes(include="number").columns.difference(["seed"])
    grouped = metrics.groupby(group_columns, dropna=False)
    mean = grouped[list(numeric)].mean().add_suffix("_mean")
    std = grouped[list(numeric)].std(ddof=1).fillna(0.0).add_suffix("_std")
    result = mean.join(std).reset_index()
    result["seed_count"] = grouped["seed"].nunique().to_numpy()
    result["data_provenance"] = SYNTHETIC_PROVENANCE
    return result.sort_values(group_columns).reset_index(drop=True)


def _runtime_rows(metrics: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "task",
        "model",
        "split_strategy",
        "seed",
        "training_seconds",
        "inference_seconds",
        "runtime_seconds",
        "test_sample_count",
        "data_provenance",
    ]
    return metrics.loc[:, [column for column in columns if column in metrics]].copy()


def validate_full_artifacts(
    directory: str | Path,
    required_soc_models: set[str] | None = None,
    required_seeds: set[int] | None = None,
) -> None:
    """Reject incomplete, non-finite, or internally inconsistent release artifacts."""
    destination = Path(directory)
    required = {
        "soc_metrics_by_seed.csv",
        "soc_metrics_summary.csv",
        "soh_metrics_by_seed.csv",
        "soh_metrics_summary.csv",
        "robustness_metrics.csv",
        "runtime_metrics.csv",
        "experiment_manifest.json",
    }
    missing = sorted(name for name in required if not (destination / name).exists())
    if missing:
        raise ValueError(f"full benchmark is missing artifacts: {', '.join(missing)}")
    soc = pd.read_csv(destination / "soc_metrics_by_seed.csv")
    soh = pd.read_csv(destination / "soh_metrics_by_seed.csv")
    soc_summary = pd.read_csv(destination / "soc_metrics_summary.csv")
    soh_summary = pd.read_csv(destination / "soh_metrics_summary.csv")
    robustness = pd.read_csv(destination / "robustness_metrics.csv")
    runtime = pd.read_csv(destination / "runtime_metrics.csv")
    if soc.empty or soh.empty:
        raise ValueError("full benchmark metrics cannot be empty")
    for label, frame in (("SOC", soc), ("SOH", soh)):
        numeric = frame.select_dtypes(include="number")
        if not np.isfinite(numeric.to_numpy(dtype=float)).all():
            raise ValueError(f"{label} metrics contain non-finite values")
    for label, summary in (("SOC", soc_summary), ("SOH", soh_summary)):
        required_summary = {"model", "split_strategy", "rmse_mean", "rmse_std", "seed_count"}
        if not required_summary.issubset(summary.columns):
            raise ValueError(f"{label} summary does not match the release schema")
        if not np.isfinite(summary.select_dtypes(include="number").to_numpy(dtype=float)).all():
            raise ValueError(f"{label} summary contains non-finite values")
    if robustness.empty or not {"scenario", "model", "seed", "rmse"}.issubset(robustness.columns):
        raise ValueError("robustness results are missing required scenario/model rows")
    if runtime.empty or not {"training_seconds", "inference_seconds"}.issubset(runtime.columns):
        raise ValueError("runtime results are incomplete")
    if required_soc_models and not required_soc_models.issubset(set(soc["model"])):
        missing_models = sorted(required_soc_models - set(soc["model"]))
        raise ValueError(f"full SOC results are missing models: {', '.join(missing_models)}")
    if required_seeds:
        for label, frame in (("SOC", soc), ("SOH", soh)):
            observed = set(frame["seed"].astype(int))
            if not required_seeds.issubset(observed):
                raise ValueError(f"full {label} results are missing one or more requested seeds")
    manifest = json.loads((destination / "experiment_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("data_provenance") != SYNTHETIC_PROVENANCE:
        raise ValueError("full benchmark manifest does not disclose synthetic provenance")


def _release_robustness(config: ExperimentConfig) -> pd.DataFrame:
    """Run controlled SOC shifts under the held-out-cell generalization protocol."""
    scenarios = {
        "nominal": {},
        "increased_sensor_noise": {"noise_std": config.simulation.noise_std * 3},
        "current_sensor_bias": {"current_bias_a": 0.08},
        "voltage_sensor_bias": {"voltage_bias_v": 0.03},
        "unseen_temperature_range": {"temperature_min_c": -5.0, "temperature_max_c": 8.0},
        "unseen_degradation_rate": {"degradation_rate": config.simulation.degradation_rate * 2.5},
        "held_out_load_profile": {"profile": "pulse"},
    }
    protocol = replace(config, split=replace(config.split, strategy="held_out_cell"))
    records: list[pd.DataFrame] = []
    for scenario, updates in scenarios.items():
        scenario_config = replace(protocol, simulation=replace(protocol.simulation, **updates))
        metrics, _ = run_soc_experiment(scenario_config)
        metrics["scenario"] = scenario
        records.append(metrics)
    return pd.concat(records, ignore_index=True)


def run_full_release_benchmark(config: ExperimentConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the V1 three-seed release protocol and write compact aggregate artifacts.

    The simulator runs once. Each model then sees the same fixed partitions per split
    strategy; validation remains separate from the locked test partition.
    """
    if not config.full_benchmark:
        raise ValueError("run_full_release_benchmark requires full_benchmark: true")
    frame = simulate_battery_data(config.simulation)
    strategies = config.split_strategies or (config.split.strategy,)
    soc_records: list[pd.DataFrame] = []
    soh_records: list[pd.DataFrame] = []
    diagnostics: list[pd.DataFrame] = []
    for strategy in strategies:
        strategy_config = replace(config, split=replace(config.split, strategy=strategy))
        soc, soc_diagnostics = run_soc_experiment(strategy_config, frame)
        soh, soh_diagnostics = run_soh_experiment(strategy_config, frame)
        soc_records.append(soc)
        soh_records.append(soh)
        diagnostics.extend([soc_diagnostics, soh_diagnostics])
    soc_metrics = pd.concat(soc_records, ignore_index=True)
    soh_metrics = pd.concat(soh_records, ignore_index=True)
    robustness_metrics = _release_robustness(config)
    manifest = {
        "release": "1.0.0",
        "configuration": asdict(config),
        "git_commit_sha": _git_sha(),
        "data_provenance": SYNTHETIC_PROVENANCE,
        "partition_strategies": list(strategies),
        "model_selection": (
            "Validation partitions only; locked test metrics are never used for selection."
        ),
        "result_schema": {
            "per_seed": "model, split_strategy, seed, test_sample_count, metrics, runtimes",
            "summary": "mean and sample standard deviation grouped by model and split strategy",
        },
        "soc_models": list(config.models),
        "soh_models": ["ridge", "random_forest", "hist_gradient_boosting", "mlp"],
        "artifacts": [
            "soc_metrics_by_seed.csv",
            "soc_metrics_summary.csv",
            "soh_metrics_by_seed.csv",
            "soh_metrics_summary.csv",
            "robustness_metrics.csv",
            "runtime_metrics.csv",
            "plots/",
        ],
    }
    write_full_release_artifacts(
        config.output_dir,
        soc_metrics,
        soh_metrics,
        pd.concat(diagnostics, ignore_index=True),
        manifest,
        robustness_metrics,
    )
    validate_full_artifacts(
        config.output_dir,
        required_soc_models=set(RELEASE_SOC_MODELS),
        required_seeds=set(config.seeds),
    )
    return soc_metrics, soh_metrics
