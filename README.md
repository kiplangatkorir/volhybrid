# volhybrid-starter: GARCH + Deep Learning Volatility Forecasting

This is a small, end-to-end starter project for comparing classical GARCH-family baselines with
neural and hybrid models under a rolling (walk-forward) backtest.

The code is intentionally compact, but you can run it as-is once you supply data.

## Research framing (current)

### Core question
Do ML-augmented volatility models (LSTM/Transformer hybrids) deliver statistically significant
and economically meaningful improvements over classical GARCH-family models, especially during
volatility regime shifts?

### Testable hypotheses
- H1 (average performance): ML-enhanced models improve 1-step and multi-step volatility forecasts
  vs GARCH/EGARCH.
- H2 (regime shifts): Gains are larger during high-volatility regimes (crisis / stress).
- H3 (cross-market robustness): Improvements hold across US, India, and South Africa, not just one
  market.

## Data design (current)

### Assets / indices
- US: S&P 500 index (or SPY ETF as a proxy)
- India: NIFTY 50 (or a broad NSE index series you can reliably source)
- South Africa: FTSE/JSE All Share (or a broad JSE index series)

### Returns
Use log returns: $r_t = \ln(P_t) - \ln(P_{t-1})$
Either de-mean or allow a small mean term in the model.

### Forecast target (volatility proxy)
Pick at least one (two is stronger):
- Squared returns (noisy): $proxy_t = r_t^2$
- Realized volatility from a rolling window (more stable): $RV_t = \sum_{i=0}^{h-1} r_{t-i}^2$
  Common `h` values: 5, 10, 21 trading days. Forecast $RV_{t+1}$ or $RV_{t+h}$.

Tip: using `log(RV)` as the prediction target stabilizes scale and reduces outlier impact.

## Baselines (implemented vs planned)

### Implemented now
- GARCH(1,1), EGARCH(1,1), GJR-GARCH(1,1) via `arch`
- Student-t innovations are supported (configurable in `GarchConfig`)

### Planned (not yet in code)
- Linear regression on lagged features
- Ridge/Lasso on engineered features
- Random Forest or XGBoost on lagged returns and lagged `r^2`

## Hybrid definitions (defensible variants)

### Hybrid A: GARCH backbone + LSTM/Transformer correction
Fit GARCH/EGARCH, forecast baseline variance, then learn a multiplicative correction:
$\hat{V} = V_{garch} \cdot \exp(\hat{m})$
This preserves volatility structure while letting the network learn nonlinear effects.

### Hybrid B: Time-varying GARCH parameters (RNN-driven)
Let a recurrent model output time-varying `omega_t, alpha_t, beta_t` under constraints:
$\omega_t > 0, \alpha_t \ge 0, \beta_t \ge 0, \alpha_t + \beta_t < 1$
Then update:
$\sigma_t^2 = \omega_t + \alpha_t \cdot \epsilon_{t-1}^2 + \beta_t \cdot \sigma_{t-1}^2$

## Transformer design (avoid overfit)
- Causal masking (no peeking)
- Sequence length: 60-252 trading days
- Predict `log(RV)` or `log(sigma^2)` for stability
- Enforce positivity via `exp` or `softplus`

Minimal feature set:
- Lagged returns $r_{t-k}$ for `k=1..L`
- Lagged squared returns $r_{t-k}^2$
- Optional: lagged GARCH variance and rolling stats

## Evaluation design (implemented vs planned)

### Implemented now
- Rolling-origin backtest with periodic refits
- Metrics: RMSE on `log(V)`, QLIKE, MAE, MAPE

### Planned (not yet in code)
- RMSE/MAE on `RV` or `log(RV)` for multiple targets
- Negative log-likelihood for distributional models
- Diebold-Mariano tests for statistical significance
- Regime-based performance reporting (high/low-vol + transitions)

Regime definition options:
- Quantile-based: high-vol if `RV_t` above 80th/90th percentile
- Change-point detection on `log(RV_t)`
- External indicator (e.g., VIX for US)

## What is implemented

### Targets
- Forward realized variance over the next `H` trading days:
  $V_t = \sum_{i=1}^{H} r_{t+i}^2$
  (You can later swap in realized volatility or OHLC estimators.)

### Models
- Baselines (via `arch`):
  - GARCH(1,1)
  - EGARCH(1,1)
  - GJR-GARCH(1,1)
- Direct neural models:
  - LSTM regressor (predicts `log(V_t)`)
  - Causal Transformer regressor (predicts `log(V_t)`)
- Hybrid:
  - Multiplicative correction: $V_hat = V_garch * exp(m_hat)$
    where the NN learns residual dynamics.

### Backtest
- Rolling window training (`train_window`)
- Forecast horizons `H` are configurable (e.g., 1, 5, 22)
- Refit schedule:
  - GARCH: usually refit each step
  - Neural nets: often refit every ~22 days (monthly) to reduce compute

### Metrics
- RMSE on `log(V)` (numerically stable)
- QLIKE for variance forecasts
- MAE / MAPE (optional)

## Install (recommended: venv)

```bash
pip install -r requirements.txt
```

Note: if you do not want neural models, you can skip `torch` and only run GARCH.

## Data format

Provide a CSV like:
- `Date` (parseable date)
- `Close` (required)
- Optional: `Open`, `High`, `Low`, `Volume`
- Optional exogenous series can be merged later.

Example path:
`data/SP500.csv`

## Run an experiment

### 1) GARCH baseline (daily refit)
```bash
python -m scripts.run_experiment --csv data/SP500.csv --model egarch_t --horizon 22 --train-window 1000 --lookback 60 --refit-every 1
```

### 2) LSTM direct (monthly refit)
```bash
python -m scripts.run_experiment --csv data/SP500.csv --model lstm --horizon 22 --train-window 1000 --lookback 60 --refit-every 22 --epochs 10 --device cpu
```

### 3) Hybrid EGARCH + LSTM correction
```bash
python -m scripts.run_experiment --csv data/SP500.csv --model hybrid_egarch_lstm --horizon 22 --train-window 1000 --lookback 60 --refit-every 22 --epochs 10 --device cpu
```

Outputs:
- `results/<ticker>_<model>_H<h>.csv` with predictions and truths
- Printed metric summary

## Progress so far

- Core pipeline in place: data loading, target construction, rolling backtest, and metrics.
- Model family implemented: GARCH/EGARCH/GJR, LSTM, Transformer, and hybrid multiplicative correction.
- Synthetic data generator added (`examples/make_synthetic.py`) for quick smoke tests.
- Initial run on synthetic data completed with EGARCH at horizon 1; results saved to `results/asset_egarch_t_H1.csv`.
  - Note: EGARCH analytic forecasts from `arch` are limited to horizon 1. For longer horizons, switch models or update the forecasting method.

## Likely next steps
- Regime splits (high-vol vs low-vol) and DM tests
- Richer features (rolling stats, volume, macro or VIX-like series)
- Alternative targets (Garman-Klass / Parkinson / intraday realized variance)
- Hyperparameter sweeps and early stopping

