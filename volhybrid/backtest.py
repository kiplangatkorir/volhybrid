from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Protocol, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from .metrics import summarize_metrics


class VarianceForecaster(Protocol):
    def fit(self, returns_train: pd.Series, target_var_train: pd.Series) -> None: ...
    def predict_var(self, returns_hist: pd.Series) -> float: ...


@dataclass
class BacktestResult:
    df: pd.DataFrame
    metrics: dict


def rolling_backtest(
    returns: pd.Series,
    target_var: pd.Series,
    train_window: int,
    horizon: int,
    lookback: int,
    refit_every: int,
    model_factory: Callable[[], VarianceForecaster],
    show_progress: bool = True,
) -> BacktestResult:
    r = returns.dropna().copy()
    y = target_var.reindex(r.index)

    n = len(r)
    if n < train_window + horizon + 10:
        raise ValueError("Not enough data for requested train_window/horizon.")

    preds: List[float] = []
    trues: List[float] = []
    dates: List[pd.Timestamp] = []

    model: Optional[VarianceForecaster] = None

    start = train_window - 1
    end = n - horizon - 1  # last t with defined forward target

    it = range(start, end + 1)
    if show_progress:
        it = tqdm(it, desc="rolling_backtest", total=(end - start + 1))

    for step, t in enumerate(it):
        if (model is None) or (step % refit_every == 0):
            model = model_factory()
            train_start = t - train_window + 1
            train_end = t
            r_train = r.iloc[train_start : train_end + 1]
            y_train = y.iloc[train_start : train_end + 1]
            model.fit(r_train, y_train)

        hist_start = max(0, t - lookback + 1)
        r_hist = r.iloc[hist_start : t + 1]

        y_true = float(y.iloc[t])
        if not np.isfinite(y_true):
            continue

        y_pred = float(model.predict_var(r_hist))
        preds.append(y_pred)
        trues.append(y_true)
        dates.append(r.index[t])

    out = pd.DataFrame({"pred_var": preds, "true_var": trues}, index=pd.DatetimeIndex(dates))
    metrics = summarize_metrics(out["true_var"].values, out["pred_var"].values)
    return BacktestResult(df=out, metrics=metrics)
