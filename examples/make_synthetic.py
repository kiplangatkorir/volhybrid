from __future__ import annotations
import numpy as np
import pandas as pd

def make_synthetic_price_csv(path: str, n: int = 2500, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    # Simple stochastic volatility-ish returns
    vol = np.zeros(n)
    r = np.zeros(n)
    vol[0] = 0.01
    for t in range(1, n):
        vol[t] = 0.0001 + 0.90 * vol[t-1] + 0.08 * (r[t-1] ** 2)
        r[t] = rng.normal(0.0, np.sqrt(max(vol[t], 1e-8)))
    price = 100 * np.exp(np.cumsum(r))
    dates = pd.bdate_range("2015-01-01", periods=n)
    df = pd.DataFrame({"Date": dates, "Close": price})
    df.to_csv(path, index=False)

if __name__ == "__main__":
    make_synthetic_price_csv("synthetic.csv")
    print("Wrote synthetic.csv")
