"""Run SOC benchmark models and write SOC artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from battery_ml.config import load_config
from battery_ml.experiment import run_soc_experiment
from battery_ml.reporting import write_reports


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate SOC models on leakage-safe partitions."
    )
    parser.add_argument("--config", required=True, help="YAML experiment configuration")
    args = parser.parse_args()
    config = load_config(args.config)
    metrics, diagnostics = run_soc_experiment(config)
    write_reports(config.output_dir, soc_metrics=metrics, diagnostics=diagnostics)
    print(f"Wrote {config.output_dir}/soc_metrics.csv")


if __name__ == "__main__":
    main()
