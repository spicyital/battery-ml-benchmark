# Public-data adapter

This repository starts with generated synthetic data. A public dataset may be loaded only after its source, license, redistribution terms, and required attribution have been reviewed and recorded. No public dataset has been downloaded or validated by this repository release.

## Required input schema

The adapter should normalize raw observations to one row per sample with these fields:

| Field | Requirement |
| --- | --- |
| `timestamp_s` | Monotonic time within a cycle. |
| `cell_id`, `cycle_id`, `sample_id` | Stable identifiers for grouped splitting and time ordering. |
| `current_a`, `voltage_v`, `temperature_c` | Calibrated observed sensor values with sign convention documented. |
| `soc` | Continuous SOC target when SOC evaluation is requested. |
| `soh` | Cycle-level SOH target, with its capacity/reference definition documented. |
| metadata | Chemistry, nominal capacity, protocol, source URL, license, and preprocessing record. |

If either target is absent, the adapter must not claim to support that evaluation task. Derived SOH targets must document their reference capacity and must not be calculated from evaluation-period information unavailable at inference time.

## Leakage-safe integration

- Store raw downloads outside Git (for example, an ignored local data directory) and never commit credentials or restricted data.
- Assign complete cells and complete cycles to train, validation, and test partitions before fitting transforms or generating learned features.
- Fit scalers and imputation only on training groups.
- Use only current and past sensor samples for time-series features and windows.
- Retain an immutable provenance record containing the dataset version, license, checksum where allowed, and adapter settings.
- Keep the final test set locked until model selection is complete.

An adapter belongs in `src/battery_ml/data_loading.py` and should reject missing identifiers, non-monotonic samples, undocumented licenses, and ambiguous SOC/SOH semantics. Adding an adapter does not constitute public-data validation; a separate documented experiment is required.
