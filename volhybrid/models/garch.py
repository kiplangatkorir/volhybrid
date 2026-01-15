from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from arch import arch_model

from ..config import GarchConfig


class ArchGarchForecaster:
    """
    Forecasts integrated forward variance over next H days by summing variance forecasts (1..H).
    Uses `arch` for estimation.
    """

    def __init__(self, horizon: int, cfg: GarchConfig):
        self.horizon = int(horizon)
        self.cfg = cfg
        self._res = None

    def fit(self, returns_train: pd.Series, target_var_train: pd.Series) -> None:
        r = (returns_train.astype(float) * 100.0).dropna()
        am = arch_model(
            r,
            mean=self.cfg.mean,
            vol=self.cfg.vol,
            p=self.cfg.p,
            o=self.cfg.o,
            q=self.cfg.q,
            dist=self.cfg.dist,
        )
        self._res = am.fit(disp="off")

    def predict_var(self, returns_hist: pd.Series) -> float:
        if self._res is None:
            raise RuntimeError("Call fit() before predict_var().")
        f = self._res.forecast(horizon=self.horizon, reindex=False)
        daily_var_pct = f.variance.values[-1, : self.horizon]   # (% returns)^2
        daily_var = daily_var_pct / (100.0 ** 2)                # back to fraction^2
        return float(np.sum(daily_var))
