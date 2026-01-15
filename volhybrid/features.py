from __future__ import annotations

import pandas as pd
import numpy as np


def build_features_basic(returns: pd.Series, extra: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Minimal feature set:
      - r
      - r^2
      - rolling mean/var (optional light memory)
      - (optionally) merge extra exogenous features already aligned by Date
    """
    r = returns.astype(float)
    df = pd.DataFrame(
        {
            "r": r,
            "r2": r ** 2,
            "r_mean_5": r.rolling(5).mean(),
            "r2_mean_5": (r ** 2).rolling(5).mean(),
            "r2_mean_21": (r ** 2).rolling(21).mean(),
        },
        index=returns.index,
    )

    if extra is not None:
        df = df.join(extra, how="left")

    return df
