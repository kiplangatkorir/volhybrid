# Unified Project Checklist

This table aligns open GitHub issues with items from IMPLEMENTATION_STATUS.md to provide a comprehensive view of all planned work.

| Key | Work Item | Source | Scope | Priority | Suggested Owner Role | Suggested Due Date | Notes |
|-----|-----------|--------|-------|----------|---------------------|-------------------|-------|
| #1 | Add regime evaluation + DM tests | Issues | Evaluation | P0 | Validation Lead | 2026-01-30 | Implements regime-aware metrics and Diebold-Mariano tests; critical for core paper claim about performance during stress; extend `volhybrid/metrics.py` |
| #2 | Implement simple ML baselines (linear/ridge + RF/XGBoost) | Issues | Models | P1 | ML Lead-Transformer | 2026-02-06 | Adds OLS, Ridge, Lasso, Random Forest, XGBoost baselines; reviewer-expected comparisons; integrate via `--model` flag in `scripts/run_experiment.py` |
| #3 | Fix multi-step GARCH forecasts (H>1) | Issues | Models | P0 | Quant Lead | 2026-01-30 | EGARCH/GJR fail for horizons H>1; implement simulation-based multi-step forecast; blocks multi-horizon experiments; fix in `volhybrid/models/garch.py` |
| #4 | Add data pipeline + datasets (S&P 500 / NIFTY / JSE) | Issues | Data | P0 | Data Lead | 2026-01-30 | Reproducible data loaders for three target markets; critical for paper replicability; create new `data/loaders/` module with market-specific handlers |
| NEW-1 | MAE/RMSE on RV or log(RV) for multiple targets | IMPLEMENTATION_STATUS | Evaluation | P1 | Validation Lead | 2026-02-06 | Extend metrics beyond QLIKE; reviewer-expected alternative loss functions; add to `volhybrid/metrics.py` |
| NEW-2 | Negative log-likelihood for distributional models | IMPLEMENTATION_STATUS | Evaluation | P2 | Quant Lead | 2026-02-22 | Probabilistic forecast evaluation; extension for advanced model comparison; implement in `volhybrid/metrics.py` |
| NEW-3 | Alternative Targets (RV variants, OHLC estimators) | IMPLEMENTATION_STATUS | Features | P2 | Data Lead | 2026-02-22 | Realized volatility variants (rolling RV, Garman-Klass, Parkinson); extension beyond sum-of-squared-returns; extend `volhybrid/features.py` or create new module |
| NEW-4 | Advanced Hybrid: Time-varying GARCH parameters | IMPLEMENTATION_STATUS | Models | P2 | ML Lead-LSTM | 2026-02-22 | RNN/Transformer-driven GARCH parameters; research extension beyond multiplicative correction; extend `volhybrid/models/hybrid.py` |
| NEW-5 | Feature Expansion: Macro/volume/VIX indices | IMPLEMENTATION_STATUS | Features | P2 | Data Lead | 2026-02-22 | External volatility indices and macro features; extension for exogenous signal integration; extend `volhybrid/features.py` or create new module |

## Priority Definitions

- **P0**: Blocks core paper claims (regime evaluation, DM tests, multi-horizon GARCH, reproducible data)
- **P1**: Reviewer-expected baselines and standard metrics (ML baselines, alternative loss functions)
- **P2**: Extensions and advanced features (alternative targets, exogenous features, advanced hybrids)

## Due Date Calculation

- **P0**: Within 7 days from 2026-01-23 → 2026-01-30
- **P1**: Within 14 days from 2026-01-23 → 2026-02-06
- **P2**: Within 30 days from 2026-01-23 → 2026-02-22

## Summary

- **Total Items**: 9
- **From GitHub Issues (OPEN)**: 4
- **From IMPLEMENTATION_STATUS (NEW)**: 5
- **P0 Items**: 3 (regime eval + DM, GARCH H>1 fix, data pipeline)
- **P1 Items**: 2 (ML baselines, alternative metrics)
- **P2 Items**: 4 (NLL, alternative targets, advanced hybrid, feature expansion)
