# Copy-ready article results

| Model | RMSE | test R² | BMR | BMR_tol | BMR_Z |
|---|---:|---:|---:|---:|---:|
| Martingale | 0.023364 | -0.0007 | 100.0% | 50.2% | 100.0% |
| Median Quantile | 0.023350 | 0.0006 | 97.5% | 46.5% | 100.0% |
| HuPin LSTM | 0.023614 | -0.0222 | 79.9% | 34.4% | 100.0% |

Common test panel: 1,575 security-days (315 dates for each of five UK-listed energy securities), 2024-04-22 to 2025-07-18. For the article, BMR_tol uses a pre-specified one-percentage-point materiality threshold (epsilon = 0.01), replacing the MSc model-dependent validation-RMSE rule. BMR_Z uses one ticker-specific, train-only Z = 2 threshold shared by all models. The BMR family is interpreted for a long position.
