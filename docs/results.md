# Results

All values below are generated from synthetic data; they are not real-battery accuracy claims. The committed quick benchmark includes Ridge, Random Forest, Histogram Gradient Boosting, MLP, NARX, coulomb counting, OCV lookup, and EKF. CNN and GRU are represented in the separate Version 1 full-release artifacts under `results/full/`, but not in these committed quick results.

Lowest quick SOC RMSE: **ridge**. Lowest quick SOH RMSE: **ridge**.

## SOC quick benchmark

| Model | RMSE | MAE | R2 | Runtime (s) | Data |
| --- | ---: | ---: | ---: | ---: | --- |
| ridge | 0.010643 | 0.008990 | 0.694956 | 0.002441 | synthetic data generated inside this project |
| random_forest | 0.123091 | 0.112844 | -39.804726 | 0.129591 | synthetic data generated inside this project |
| hist_gradient_boosting | 0.116671 | 0.106213 | -35.659184 | 0.727694 | synthetic data generated inside this project |
| mlp | 0.106367 | 0.082255 | -29.469935 | 0.111231 | synthetic data generated inside this project |
| narx | 0.029382 | 0.025676 | -1.324953 | 0.176594 | synthetic data generated inside this project |
| coulomb_counting | 0.150191 | 0.147377 | -59.749977 | 0.011158 | synthetic data generated inside this project |
| ocv_lookup | 0.025463 | 0.019725 | -0.746145 | 0.000069 | synthetic data generated inside this project |
| ekf | 0.023860 | 0.018041 | -0.533235 | 0.013118 | synthetic data generated inside this project |

## SOH quick benchmark

| Model | RMSE | MAE | R2 | Runtime (s) | Data |
| --- | ---: | ---: | ---: | ---: | --- |
| ridge | 0.007497 | 0.006559 | 0.745378 | 0.000685 | synthetic data generated inside this project |
| random_forest | 0.008533 | 0.007967 | 0.670179 | 0.026517 | synthetic data generated inside this project |
| hist_gradient_boosting | 0.015887 | 0.013475 | -0.143324 | 0.053314 | synthetic data generated inside this project |
| mlp | 0.061842 | 0.040359 | -16.323479 | 0.023521 | synthetic data generated inside this project |
