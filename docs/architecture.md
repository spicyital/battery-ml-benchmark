# Architecture

The benchmark uses a four-stage pipeline: synthetic simulation or a documented permitted-public adapter; partition assignment by complete cell/cycle; causal preprocessing and modeling; artifact/report generation. `experiment.py` owns orchestration, while individual modules are independently testable.

ML estimators live in `models/`. Engineering methods in `baselines/` are deliberately isolated and labelled as comparison methods so the project remains ML-first. Sequence inputs are strictly historical: each CNN/GRU target occurs one sample after its final input row.
