from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

from ..config import GarchConfig, TorchConfig
from ..features import build_features_basic
from ..utils import EPS
from .garch import ArchGarchForecaster
from .torch_models import SequenceDataset, LSTMRegressor, CausalTransformerRegressor


class HybridMultiplicativeCorrector:
    """
    Hybrid A:
      V_hat = V_garch * exp(m_hat)
    where m_t is learned as:
      m_t = log(V_true) - log(V_garch)
    """

    def __init__(self, horizon: int, lookback: int, garch_cfg: GarchConfig, torch_cfg: TorchConfig):
        self.horizon = int(horizon)
        self.lookback = int(lookback)
        self.garch_cfg = garch_cfg
        self.torch_cfg = torch_cfg

        self.garch = ArchGarchForecaster(horizon=self.horizon, cfg=garch_cfg)
        self.scaler = StandardScaler()
        self.model: nn.Module | None = None

    def _init_model(self, n_features: int) -> nn.Module:
        if self.torch_cfg.model_type == "lstm":
            return LSTMRegressor(n_features, hidden=self.torch_cfg.hidden, layers=self.torch_cfg.layers, dropout=self.torch_cfg.dropout)
        if self.torch_cfg.model_type == "transformer":
            return CausalTransformerRegressor(n_features, d_model=self.torch_cfg.hidden, layers=self.torch_cfg.layers, dropout=self.torch_cfg.dropout)
        raise ValueError(f"Unknown model_type: {self.torch_cfg.model_type}")

    def _garch_integrated_var_series(self, returns_train: pd.Series) -> pd.Series:
        if self.garch._res is None:
            raise RuntimeError("GARCH must be fit first.")
        f = self.garch._res.forecast(horizon=self.horizon, reindex=False)
        var = f.variance / (100.0 ** 2)
        integrated = var.iloc[:, : self.horizon].sum(axis=1)
        return integrated.reindex(returns_train.index)

    def fit(self, returns_train: pd.Series, target_var_train: pd.Series) -> None:
        # Fit baseline GARCH/EGARCH/GJR
        self.garch.fit(returns_train, target_var_train)

        Xdf = build_features_basic(returns_train).dropna()
        Xdf["garch_int_var"] = self._garch_integrated_var_series(returns_train)
        Xdf = Xdf.dropna()

        y = target_var_train.reindex(Xdf.index).astype(float)
        mask = np.isfinite(y.values)
        Xdf = Xdf.loc[mask]
        y = y.loc[mask]

        g = Xdf["garch_int_var"].values.astype(float)
        m = np.log(y.values + EPS) - np.log(g + EPS)

        X = Xdf.values.astype(float)
        self.scaler.fit(X)
        Xs = self.scaler.transform(X)

        ds = SequenceDataset(Xs, m, lookback=self.lookback, horizon=self.horizon)
        dl = DataLoader(ds, batch_size=self.torch_cfg.batch_size, shuffle=True)

        device = torch.device(self.torch_cfg.device)
        self.model = self._init_model(n_features=Xs.shape[1]).to(device)

        opt = torch.optim.Adam(self.model.parameters(), lr=self.torch_cfg.lr)
        loss_fn = nn.MSELoss()

        self.model.train()
        for _ in range(self.torch_cfg.epochs):
            for xb, yb in dl:
                xb = xb.to(device)
                yb = yb.to(device)
                opt.zero_grad()
                pred = self.model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()

    def predict_var(self, returns_hist: pd.Series) -> float:
        if self.model is None:
            raise RuntimeError("Call fit() before predict_var().")

        v_garch = self.garch.predict_var(returns_hist)

        Xdf = build_features_basic(returns_hist).dropna()
        # At prediction time, we only have one baseline forecast for t, so treat it as constant feature.
        Xdf["garch_int_var"] = v_garch
        X = Xdf.values.astype(float)
        Xs = self.scaler.transform(X)

        if len(Xs) < self.lookback:
            raise ValueError("Not enough history for lookback.")

        x_seq = Xs[-self.lookback :]

        device = torch.device(self.torch_cfg.device)
        self.model.eval()
        with torch.no_grad():
            xb = torch.tensor(x_seq[None, :, :], dtype=torch.float32, device=device)
            m_hat = float(self.model(xb).cpu().numpy().reshape(-1)[0])

        return float(v_garch * np.exp(m_hat))
