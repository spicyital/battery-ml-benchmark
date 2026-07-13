"""Execute the complete synthetic SOC/SOH benchmark."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from battery_ml.config import load_config
from battery_ml.experiment import run_full_benchmark
from battery_ml.robustness import run_robustness


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the complete SOC and SOH battery ML benchmark."
    )
    parser.add_argument("--config", required=True, help="YAML experiment configuration")
    args = parser.parse_args()
    config = load_config(args.config)
    run_full_benchmark(config)
    run_robustness(config)


if __name__ == "__main__":
    main()
