from __future__ import annotations

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

from ..config import TorchConfig
from ..features import build_features_basic
from ..utils import EPS


class SequenceDataset(Dataset):
    """Builds (X_seq, y_t) with no lookahead leakage inside the train slice."""

    def __init__(self, X: np.ndarray, y: np.ndarray, lookback: int, horizon: int):
        self.X = X
        self.y = y
        self.lookback = int(lookback)
        self.horizon = int(horizon)

        N = len(y)
        self.min_t = self.lookback - 1
        self.max_t = N - 1 - self.horizon
        if self.max_t < self.min_t:
            raise ValueError("Train slice too small for lookback/horizon.")

        self.idxs = np.arange(self.min_t, self.max_t + 1)

    def __len__(self) -> int:
        return len(self.idxs)

    def __getitem__(self, i: int):
        t = int(self.idxs[i])
        x_seq = self.X[t - self.lookback + 1 : t + 1]  # [L, F]
        y_t = self.y[t]
        return torch.tensor(x_seq, dtype=torch.float32), torch.tensor(y_t, dtype=torch.float32)


class LSTMRegressor(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=layers,
            dropout=dropout if layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)


class CausalTransformerRegressor(nn.Module):
    def __init__(self, n_features: int, d_model: int = 64, nhead: int = 4, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.in_proj = nn.Linear(n_features, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )

    @staticmethod
    def causal_mask(L: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(L, L, device=device, dtype=torch.bool), diagonal=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        z = self.in_proj(x)
        mask = self.causal_mask(L, z.device)
        z = self.encoder(z, mask=mask)
        last = z[:, -1, :]
        return self.head(last).squeeze(-1)


class TorchDirectForecaster:
    """
    Direct model: predict log(V_t) then exponentiate to get V_hat.
    Features are built from returns only by default; extend build_features_basic() as needed.
    """

    def __init__(self, horizon: int, lookback: int, cfg: TorchConfig):
        self.horizon = int(horizon)
        self.lookback = int(lookback)
        self.cfg = cfg

        self.scaler = StandardScaler()
        self.model: nn.Module | None = None

    def _init_model(self, n_features: int) -> nn.Module:
        if self.cfg.model_type == "lstm":
            return LSTMRegressor(n_features, hidden=self.cfg.hidden, layers=self.cfg.layers, dropout=self.cfg.dropout)
        if self.cfg.model_type == "transformer":
            return CausalTransformerRegressor(n_features, d_model=self.cfg.hidden, layers=self.cfg.layers, dropout=self.cfg.dropout)
        raise ValueError(f"Unknown model_type: {self.cfg.model_type}")

    def fit(self, returns_train: pd.Series, target_var_train: pd.Series) -> None:
        Xdf = build_features_basic(returns_train).dropna()
        y = target_var_train.reindex(Xdf.index).astype(float)

        mask = np.isfinite(y.values)
        Xdf = Xdf.loc[mask]
        y = y.loc[mask]

        X = Xdf.values.astype(float)
        y_log = np.log(y.values + EPS)

        self.scaler.fit(X)
        Xs = self.scaler.transform(X)

        ds = SequenceDataset(Xs, y_log, lookback=self.lookback, horizon=self.horizon)
        dl = DataLoader(ds, batch_size=self.cfg.batch_size, shuffle=True)

        device = torch.device(self.cfg.device)
        self.model = self._init_model(n_features=Xs.shape[1]).to(device)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        loss_fn = nn.MSELoss()

        self.model.train()
        for _ in range(self.cfg.epochs):
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

        Xdf = build_features_basic(returns_hist).dropna()
        X = Xdf.values.astype(float)
        Xs = self.scaler.transform(X)

        if len(Xs) < self.lookback:
            raise ValueError("Not enough history for lookback.")

        x_seq = Xs[-self.lookback :]

        device = torch.device(self.cfg.device)
        self.model.eval()
        with torch.no_grad():
            xb = torch.tensor(x_seq[None, :, :], dtype=torch.float32, device=device)
            y_log = float(self.model(xb).cpu().numpy().reshape(-1)[0])

        return float(np.exp(y_log))
