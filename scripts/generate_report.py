"""Regenerate Markdown and plots from saved CSV artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from battery_ml.reporting import write_reports


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate plots and Markdown report from benchmark results."
    )
    parser.add_argument("--results", required=True, help="Results directory")
    args = parser.parse_args()
    directory = Path(args.results)
    soc = (
        pd.read_csv(directory / "soc_metrics.csv")
        if (directory / "soc_metrics.csv").exists()
        else pd.DataFrame()
    )
    soh = (
        pd.read_csv(directory / "soh_metrics.csv")
        if (directory / "soh_metrics.csv").exists()
        else pd.DataFrame()
    )
    diagnostics = (
        pd.read_csv(directory / "prediction_diagnostics.csv")
        if (directory / "prediction_diagnostics.csv").exists()
        else pd.DataFrame()
    )
    manifest = (
        json.loads((directory / "experiment_manifest.json").read_text())
        if (directory / "experiment_manifest.json").exists()
        else {}
    )
    write_reports(directory, soc, soh, diagnostics, manifest)


if __name__ == "__main__":
    main()
