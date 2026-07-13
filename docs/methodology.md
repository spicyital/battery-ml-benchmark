# Methodology

SOC is a row-level regression task over observable current, voltage, temperature, elapsed time, causal lags, rolling statistics, derivatives, charge state, cumulative ampere-hours, and temperature interactions. Simulator-only true SOC/SOH, capacity, resistance, and cycle-start SOC are excluded from ML inputs. SOH is a cycle-level regression task over estimated discharge capacity, voltage-curve summaries, current/temperature summaries, a resistance proxy, durations, and historical-cycle context.

CNN/GRU early stopping uses the configured validation partition. Other quick models use fixed, documented settings and do not make a random internal validation split; no hyperparameter choice is made from the final test partition. The quick configuration records one deterministic seed; full configurations record multiple seeds as separate rows for later mean/std aggregation.
