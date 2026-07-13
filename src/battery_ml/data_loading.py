"""Safe local loading plus documented-public-source adapter metadata."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_synthetic_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def public_dataset_adapter(
    source_url: str, license_name: str, local_path: str | Path
) -> pd.DataFrame:
    """Load an already-obtained public file only when source and license are documented."""
    if not source_url or not license_name:
        raise ValueError("public datasets require documented source_url and license_name")
    return pd.read_csv(local_path)
