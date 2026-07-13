# Architecture

The benchmark uses a four-stage pipeline: synthetic simulation or a documented permitted-public adapter; partition assignment by complete cell/cycle; causal preprocessing and modeling; artifact/report generation. `experiment.py` owns orchestration, while individual modules are independently testable.

ML estimators live in `models/`. Engineering methods in `baselines/` are deliberately isolated and labelled as comparison methods so the project remains ML-first. Sequence inputs are strictly historical: each CNN/GRU target occurs one sample after its final input row.

`configs/quick.yaml` is the fast CI path. `configs/full_benchmark.yaml` uses the same architecture for three seeds, complete held-out-cell and chronological held-out-cycle protocols, and compact CPU-only CNN/GRU training. Full aggregate artifacts are isolated in `results/full/`; raw synthetic trajectories and temporary checkpoints are ignored.
