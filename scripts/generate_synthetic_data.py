"""Generate locally synthetic battery trajectories."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from battery_ml.config import load_config
from battery_ml.simulation import simulate_battery_data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate deterministic synthetic battery benchmark data."
    )
    parser.add_argument("--config", required=True, help="YAML experiment configuration")
    parser.add_argument(
        "--output", default="data/synthetic/battery_cycles.csv", help="CSV destination"
    )
    args = parser.parse_args()
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    simulate_battery_data(load_config(args.config).simulation).to_csv(destination, index=False)
    print(destination)


if __name__ == "__main__":
    main()
