from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class BacktestConfig:
    horizon: int = 22          # H
    lookback: int = 60         # L (for NN sequences)
    train_window: int = 1000   # W
    refit_every: int = 1       # refit frequency


@dataclass(frozen=True)
class GarchConfig:
    vol: str = "GARCH"         # "GARCH", "EGARCH", "GARCH" with o>0 for GJR
    p: int = 1
    o: int = 0
    q: int = 1
    dist: str = "normal"       # "normal" or "t"
    mean: str = "Zero"         # "Zero" or "Constant"


@dataclass(frozen=True)
class TorchConfig:
    model_type: str = "lstm"   # "lstm" or "transformer"
    hidden: int = 64
    layers: int = 2
    dropout: float = 0.1
    lr: float = 1e-3
    batch_size: int = 128
    epochs: int = 10
    device: str = "cpu"
