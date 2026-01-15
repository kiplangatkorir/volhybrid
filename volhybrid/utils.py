from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-12


def read_price_csv(path: str) -> pd.DataFrame:
    """Read a price CSV with at least Date, Close. Returns a DataFrame indexed by Date."""
    df = pd.read_csv(path)
    if "Date" not in df.columns:
        raise ValueError("CSV must include a 'Date' column.")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").set_index("Date")

    if "Close" not in df.columns:
        raise ValueError("CSV must include a 'Close' column.")
    return df


def log_returns(close: pd.Series) -> pd.Series:
    r = np.log(close.astype(float)).diff()
    return r.dropna()


def forward_realized_variance(returns: pd.Series, horizon: int) -> pd.Series:
    """
    Forward realized variance target:
      V_t = sum_{i=1..h} r_{t+i}^2
    Undefined for last h obs.
    """
    r2 = (returns.astype(float) ** 2).values
    n = len(r2)
    out = np.full(n, np.nan, dtype=float)

    cs = np.concatenate([[0.0], np.cumsum(r2)])  # length n+1
    if n > horizon:
        out[: n - horizon] = cs[horizon + 1 : n + 1] - cs[1 : n - horizon + 1]

    return pd.Series(out, index=returns.index, name=f"fwd_var_h{horizon}")


def rmse_log_var(y_true_var: np.ndarray, y_pred_var: np.ndarray) -> float:
    yt = np.log(y_true_var + EPS)
    yp = np.log(y_pred_var + EPS)
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def qlike(y_true_var: np.ndarray, y_pred_var: np.ndarray) -> float:
    pred = y_pred_var + EPS
    true = y_true_var + EPS
    return float(np.mean(np.log(pred) + (true / pred)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.maximum(np.abs(y_true), EPS)
    return float(np.mean(np.abs((y_true - y_pred) / denom)))
