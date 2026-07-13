"""Artifact, plot, and Markdown report generation without external tracking."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


def _plot_line(frame: pd.DataFrame, x: str, y: str, path: Path, title: str, ylabel: str) -> None:
    plt.figure(figsize=(8, 4))
    for name, group in frame.groupby("model"):
        plt.plot(group[x], group[y], label=str(name))
    plt.title(title)
    plt.xlabel(x.replace("_", " "))
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def write_plots(
    soc_metrics: pd.DataFrame,
    soh_metrics: pd.DataFrame,
    diagnostics: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Save separate pragmatic figures for portfolio inspection."""
    if not diagnostics.empty:
        soc = diagnostics.loc[diagnostics.task == "soc"]
        if not soc.empty:
            first = soc.loc[soc.model == soc.model.iloc[0]]
            plt.figure(figsize=(8, 4))
            plt.plot(first.index, first.y_true, label="true SOC")
            plt.plot(first.index, first.y_pred, label="predicted SOC")
            plt.title(f"SOC true versus predicted ({first.model.iloc[0]})")
            plt.xlabel("test observation")
            plt.ylabel("SOC")
            plt.legend()
            plt.tight_layout()
            plt.savefig(output_dir / "soc_true_vs_predicted.png", dpi=140)
            plt.close()
            plt.figure(figsize=(8, 4))
            plt.plot(first.index, first.y_pred - first.y_true)
            plt.title("SOC residuals over time")
            plt.xlabel("test observation")
            plt.ylabel("prediction error")
            plt.tight_layout()
            plt.savefig(output_dir / "soc_residuals_over_time.png", dpi=140)
            plt.close()
            if "temperature_c" in first:
                plt.figure(figsize=(8, 4))
                plt.scatter(first.temperature_c, first.absolute_error)
                plt.title("SOC error by temperature")
                plt.xlabel("temperature (C)")
                plt.ylabel("absolute error")
                plt.tight_layout()
                plt.savefig(output_dir / "soc_error_by_temperature.png", dpi=140)
                plt.close()
            plt.figure(figsize=(8, 4))
            first.assign(
                soc_band=pd.cut(first.y_true, [0, 0.2, 0.8, 1.0], include_lowest=True)
            ).groupby("soc_band", observed=True)["absolute_error"].mean().plot(kind="bar")
            plt.title("SOC error by range")
            plt.xlabel("SOC range")
            plt.ylabel("mean absolute error")
            plt.tight_layout()
            plt.savefig(output_dir / "soc_error_by_range.png", dpi=140)
            plt.close()
        soh = diagnostics.loc[diagnostics.task == "soh"]
        if not soh.empty:
            first_soh = soh.loc[soh.model == soh.model.iloc[0]]
            plt.figure(figsize=(8, 4))
            plt.plot(first_soh.cycle_id, first_soh.y_true, marker="o", label="true SOH")
            plt.plot(first_soh.cycle_id, first_soh.y_pred, marker="x", label="predicted SOH")
            plt.title(f"SOH true versus predicted ({first_soh.model.iloc[0]})")
            plt.xlabel("cycle")
            plt.ylabel("SOH")
            plt.legend()
            plt.tight_layout()
            plt.savefig(output_dir / "soh_true_vs_predicted.png", dpi=140)
            plt.close()
    if not soc_metrics.empty:
        _plot_line(
            soc_metrics.sort_values("rmse"),
            "model",
            "rmse",
            output_dir / "model_rmse_comparison.png",
            "SOC RMSE comparison",
            "RMSE",
        )
        _plot_line(
            soc_metrics.sort_values("r2"),
            "model",
            "r2",
            output_dir / "model_r2_comparison.png",
            "SOC R2 comparison",
            "R2",
        )
        _plot_line(
            soc_metrics.sort_values("runtime_seconds"),
            "model",
            "runtime_seconds",
            output_dir / "runtime_comparison.png",
            "Runtime comparison",
            "seconds",
        )
        if "cell_id" in diagnostics:
            per_cell = (
                diagnostics.loc[diagnostics.task == "soc"]
                .groupby(["model", "cell_id"], as_index=False)["absolute_error"]
                .mean()
            )
            _plot_line(
                per_cell,
                "cell_id",
                "absolute_error",
                output_dir / "per_cell_performance.png",
                "Per-cell SOC error",
                "MAE",
            )
    if not soh_metrics.empty:
        _plot_line(
            soh_metrics.sort_values("rmse"),
            "model",
            "rmse",
            output_dir / "soh_error_by_cycle.png",
            "SOH RMSE comparison",
            "RMSE",
        )
    histories = sorted(output_dir.glob("*.history.json"))
    for history_path in histories:
        history = pd.DataFrame(json.loads(history_path.read_text(encoding="utf-8")))
        if not history.empty:
            plt.figure(figsize=(8, 4))
            plt.plot(history.epoch, history.training_loss, label="training loss")
            plt.plot(history.epoch, history.validation_loss, label="validation loss")
            plt.title(f"Training and validation loss ({history_path.stem})")
            plt.xlabel("epoch")
            plt.ylabel("MSE")
            plt.legend()
            plt.tight_layout()
            plt.savefig(output_dir / f"{history_path.stem}_loss.png", dpi=140)
            plt.close()


def write_reports(
    output_dir: str | Path,
    soc_metrics: pd.DataFrame | None = None,
    soh_metrics: pd.DataFrame | None = None,
    diagnostics: pd.DataFrame | None = None,
    manifest: dict[str, Any] | None = None,
) -> None:
    """Persist all required portable artifacts and concise interpretation documents."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    soc_metrics = pd.DataFrame() if soc_metrics is None else soc_metrics
    soh_metrics = pd.DataFrame() if soh_metrics is None else soh_metrics
    diagnostics = pd.DataFrame() if diagnostics is None else diagnostics
    soc_metrics.to_csv(directory / "soc_metrics.csv", index=False)
    soh_metrics.to_csv(directory / "soh_metrics.csv", index=False)
    diagnostics.to_csv(directory / "prediction_diagnostics.csv", index=False)
    with (directory / "experiment_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest or {}, handle, indent=2, default=str)
    write_plots(soc_metrics, soh_metrics, diagnostics, directory)
    robustness_path = directory / "robustness_metrics.csv"
    if robustness_path.exists():
        robustness = pd.read_csv(robustness_path)
        if not robustness.empty:
            plt.figure(figsize=(9, 4))
            for model, group in robustness.groupby("model"):
                plt.plot(group["scenario"], group["rmse"], marker="o", label=str(model))
            plt.title("Robustness comparison")
            plt.xlabel("scenario")
            plt.ylabel("SOC RMSE")
            plt.xticks(rotation=30, ha="right")
            plt.legend()
            plt.tight_layout()
            plt.savefig(directory / "robustness_comparison.png", dpi=140)
            plt.close()
    docs = directory.parent / "docs"
    docs.mkdir(exist_ok=True)
    best_soc = (
        "No SOC model was evaluated."
        if soc_metrics.empty
        else soc_metrics.loc[soc_metrics.rmse.idxmin(), "model"]
    )
    best_soh = (
        "No SOH model was evaluated."
        if soh_metrics.empty
        else soh_metrics.loc[soh_metrics.rmse.idxmin(), "model"]
    )

    def metric_table(metrics: pd.DataFrame) -> str:
        if metrics.empty:
            return "No results were generated."
        lines = [
            "| Model | RMSE | MAE | R2 | Runtime (s) | Data |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
        for _, row in metrics.iterrows():
            lines.append(
                "| {model} | {rmse:.6f} | {mae:.6f} | {r2:.6f} | {runtime:.6f} | {data} |".format(
                    model=row["model"],
                    rmse=row["rmse"],
                    mae=row["mae"],
                    r2=row["r2"],
                    runtime=row["runtime_seconds"],
                    data=row.get("data_provenance", "synthetic"),
                )
            )
        return "\n".join(lines)

    (docs / "results.md").write_text(
        "# Results\n\n"
        "All values below are generated from synthetic data; they are not real-battery "
        "accuracy claims. The committed quick benchmark includes Ridge, Random Forest, "
        "Histogram Gradient Boosting, MLP, NARX, coulomb counting, OCV lookup, and EKF. "
        "CNN and GRU are supported in the full SOC configuration but are not represented "
        "in these committed quick results.\n\n"
        f"Lowest quick SOC RMSE: **{best_soc}**. Lowest quick SOH RMSE: **{best_soh}**.\n\n"
        "## SOC quick benchmark\n\n"
        f"{metric_table(soc_metrics)}\n\n"
        "## SOH quick benchmark\n\n"
        f"{metric_table(soh_metrics)}\n",
        encoding="utf-8",
    )
    (docs / "model-card.md").write_text(
        "# Model Card\n\n"
        "## Intended use\nBenchmark leakage-safe SOC/SOH estimation techniques on "
        "synthetic or explicitly permitted public data.\n\n"
        "## Non-intended use\nNot a battery-management-system controller or a "
        "safety-certified real-world estimator.\n\n"
        "## Training data\nSynthetic multicell, multicycle trajectories generated locally.\n\n"
        "## Evaluation procedure\nCell- or cycle-held-out partitions, causal features, "
        "training-only transforms, and robustness scenarios.\n\n"
        "## Performance\nSee the generated SOC/SOH metric CSVs for locked synthetic-test "
        "metrics and runtime.\n\n"
        "## Limitations and safety considerations\nA synthetic-to-real domain gap remains; "
        "validate against licensed representative measurements, calibration, uncertainty "
        "analysis, and safety engineering before deployment.\n\n"
        "## Synthetic-to-real domain gap\nSynthetic trajectories omit many pack, ageing, "
        "and sensor effects; results cannot be used as real-world accuracy claims.\n",
        encoding="utf-8",
    )
