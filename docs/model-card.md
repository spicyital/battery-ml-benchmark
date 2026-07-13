# Model Card

## Intended use
Benchmark leakage-safe SOC/SOH estimation techniques on synthetic or explicitly permitted public data.

## Non-intended use
Not a battery-management-system controller or a safety-certified real-world estimator.

## Training data
Synthetic multicell, multicycle trajectories generated locally.

## Evaluation procedure
Cell- or cycle-held-out partitions, causal features, training-only transforms, and robustness scenarios.

## Performance
See the generated SOC/SOH metric CSVs for locked synthetic-test metrics and runtime.

## Limitations and safety considerations
A synthetic-to-real domain gap remains; validate against licensed representative measurements, calibration, uncertainty analysis, and safety engineering before deployment.

## Synthetic-to-real domain gap
Synthetic trajectories omit many pack, ageing, and sensor effects; results cannot be used as real-world accuracy claims.
