# Implementation Status

This document summarizes what is implemented in the codebase today versus what is still missing or
planned. It is intended as a quick, code-backed checklist to align README claims with reality.

## Implemented

### Data & Targets
- CSV ingestion with `Date` + `Close` requirements, date sorting, and indexing.
- Log-returns calculation.
- Forward realized variance target: $\(V_t = \sum_{i=1}^{H} r_{t+i}^2\)$

### Features
- Basic return-based features: `r`, `r^2`, rolling means of `r` and `r^2`, plus optional joins for
  extra aligned exogenous series.

### Models
- GARCH-family forecaster via `arch` (GARCH/EGARCH/GJR with configurable distribution/mean).
- Direct neural models: LSTM regressor and causal Transformer regressor.
- Hybrid multiplicative correction: GARCH baseline + NN residual correction.

### Backtest & Metrics
- Rolling-origin backtest with configurable `train_window`, `horizon`, `lookback`, and `refit_every`.
- Metrics: RMSE on log variance + QLIKE. (MAE/MAPE helpers exist but are not wired into summaries.)

### CLI
- `scripts/run_experiment.py` wiring for GARCH/EGARCH/GJR, LSTM, Transformer, and hybrid variants.

## Missing / Not Yet Implemented

### Additional Baselines
- Linear regression on lagged features.
- Ridge/Lasso on engineered features.
- Random Forest / XGBoost on lagged returns and lagged `r^2`.

### Evaluation Extensions
- MAE/RMSE on `RV` or `log(RV)` for multiple targets.
- Negative log-likelihood for distributional models.
- Diebold–Mariano tests.
- Regime-based performance reporting (high/low volatility splits + transitions).

### Alternative Targets
- Realized volatility variants (e.g., rolling RV), OHLC estimators (Garman–Klass, Parkinson),
  or intraday realized variance.

### Advanced Hybrid
- Time-varying GARCH parameters driven by RNN/Transformer.

### Feature Expansion
- Macro / volume / external volatility indices (e.g., VIX) feature pipelines.
