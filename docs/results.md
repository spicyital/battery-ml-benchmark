# Results

All values below are generated from synthetic data; they are not real-battery accuracy claims. The committed quick benchmark includes Ridge, Random Forest, Histogram Gradient Boosting, MLP, NARX, coulomb counting, OCV lookup, and EKF. CNN and GRU are supported in the full SOC configuration but are not represented in these committed quick results.

Lowest quick SOC RMSE: **ridge**. Lowest quick SOH RMSE: **ridge**.

## SOC quick benchmark

| Model | RMSE | MAE | R2 | Runtime (s) | Data |
| --- | ---: | ---: | ---: | ---: | --- |
| ridge | 0.010643 | 0.008990 | 0.694956 | 0.002505 | synthetic data generated inside this project |
| random_forest | 0.123091 | 0.112844 | -39.804726 | 0.114465 | synthetic data generated inside this project |
| hist_gradient_boosting | 0.116671 | 0.106213 | -35.659184 | 0.545149 | synthetic data generated inside this project |
| mlp | 0.106367 | 0.082255 | -29.469935 | 0.096804 | synthetic data generated inside this project |
| narx | 0.029382 | 0.025676 | -1.324953 | 0.144257 | synthetic data generated inside this project |
| coulomb_counting | 0.150191 | 0.147377 | -59.749977 | 0.009715 | synthetic data generated inside this project |
| ocv_lookup | 0.025463 | 0.019725 | -0.746145 | 0.000124 | synthetic data generated inside this project |
| ekf | 0.023860 | 0.018041 | -0.533235 | 0.012635 | synthetic data generated inside this project |

## SOH quick benchmark

| Model | RMSE | MAE | R2 | Runtime (s) | Data |
| --- | ---: | ---: | ---: | ---: | --- |
| ridge | 0.007497 | 0.006559 | 0.745378 | 0.000668 | synthetic data generated inside this project |
| random_forest | 0.008533 | 0.007967 | 0.670179 | 0.023877 | synthetic data generated inside this project |
| hist_gradient_boosting | 0.015887 | 0.013475 | -0.143324 | 0.049160 | synthetic data generated inside this project |
| mlp | 0.061842 | 0.040359 | -16.323479 | 0.025072 | synthetic data generated inside this project |
