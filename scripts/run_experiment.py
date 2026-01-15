from __future__ import annotations

import argparse
import os
import pandas as pd

from volhybrid.config import BacktestConfig, GarchConfig, TorchConfig
from volhybrid.utils import read_price_csv, log_returns, forward_realized_variance
from volhybrid.backtest import rolling_backtest
from volhybrid.models.garch import ArchGarchForecaster
from volhybrid.models.torch_models import TorchDirectForecaster
from volhybrid.models.hybrid import HybridMultiplicativeCorrector


def build_model_factory(args) :
    H = args.horizon
    L = args.lookback

    if args.model == "garch":
        cfg = GarchConfig(vol="GARCH", dist="normal")
        return lambda: ArchGarchForecaster(horizon=H, cfg=cfg)

    if args.model == "egarch_t":
        cfg = GarchConfig(vol="EGARCH", dist="t")
        return lambda: ArchGarchForecaster(horizon=H, cfg=cfg)

    if args.model == "gjr_t":
        # In arch: use vol='GARCH' with o>0 for threshold/asymmetry (GJR)
        cfg = GarchConfig(vol="GARCH", p=1, o=1, q=1, dist="t")
        return lambda: ArchGarchForecaster(horizon=H, cfg=cfg)

    if args.model == "lstm":
        tcfg = TorchConfig(model_type="lstm", epochs=args.epochs, device=args.device)
        return lambda: TorchDirectForecaster(horizon=H, lookback=L, cfg=tcfg)

    if args.model == "transformer":
        tcfg = TorchConfig(model_type="transformer", epochs=args.epochs, device=args.device)
        return lambda: TorchDirectForecaster(horizon=H, lookback=L, cfg=tcfg)

    if args.model == "hybrid_egarch_lstm":
        gcfg = GarchConfig(vol="EGARCH", dist="t")
        tcfg = TorchConfig(model_type="lstm", epochs=args.epochs, device=args.device)
        return lambda: HybridMultiplicativeCorrector(horizon=H, lookback=L, garch_cfg=gcfg, torch_cfg=tcfg)

    if args.model == "hybrid_egarch_transformer":
        gcfg = GarchConfig(vol="EGARCH", dist="t")
        tcfg = TorchConfig(model_type="transformer", epochs=args.epochs, device=args.device)
        return lambda: HybridMultiplicativeCorrector(horizon=H, lookback=L, garch_cfg=gcfg, torch_cfg=tcfg)

    raise ValueError(f"Unknown model: {args.model}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="Path to CSV with Date, Close (and optional OHLCV).")
    p.add_argument("--ticker", default="asset", help="Label used for output file naming.")
    p.add_argument("--model", required=True,
                   choices=["garch", "egarch_t", "gjr_t", "lstm", "transformer",
                            "hybrid_egarch_lstm", "hybrid_egarch_transformer"])
    p.add_argument("--horizon", type=int, default=22)
    p.add_argument("--train-window", type=int, default=1000)
    p.add_argument("--lookback", type=int, default=60)
    p.add_argument("--refit-every", type=int, default=1)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])

    args = p.parse_args()

    df = read_price_csv(args.csv)
    r = log_returns(df["Close"])
    y = forward_realized_variance(r, horizon=args.horizon)

    model_factory = build_model_factory(args)

    res = rolling_backtest(
        returns=r,
        target_var=y,
        train_window=args.train_window,
        horizon=args.horizon,
        lookback=args.lookback,
        refit_every=args.refit_every,
        model_factory=model_factory,
        show_progress=True,
    )

    os.makedirs("results", exist_ok=True)
    out_path = f"results/{args.ticker}_{args.model}_H{args.horizon}.csv"
    res.df.to_csv(out_path, index=True)

    print("\n=== Metrics ===")
    for k, v in res.metrics.items():
        print(f"{k}: {v:.6f}")
    print(f"Saved predictions to: {out_path}")


if __name__ == "__main__":
    main()
