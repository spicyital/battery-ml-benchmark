# Leakage prevention

- Random row-level train/test splitting is prohibited.
- Complete cells are held out for grouped evaluation; chronological evaluation holds out later complete cycles.
- Split assignment happens before scaling and feature fitting.
- Lags, rolling means/stds, derivatives, and cumulative features use current/past values only. Sequence windows stop one timestep before their target.
- Standardizers fit only training arrays.
- Latent simulator labels/parameters (`soc`, `soh`, capacity, resistance, and cycle-start SOC) are excluded from SOC ML columns.
- NARX teacher-forced output lags are used only in fitting. `predict_recursive` feeds previous predictions at inference and accepts no test target array.
- Tests assert partition disjointness, causal lags/windows, target-free sequence windows, train-only scaling, deterministic generation, hidden-state exclusion, and recursive finite predictions.
