from __future__ import annotations
import numpy as np
from .utils import rmse_log_var, qlike, mae, mape


def summarize_metrics(true_var: np.ndarray, pred_var: np.ndarray) -> dict:
    true_var = true_var.astype(float)
    pred_var = pred_var.astype(float)
    return {
        "RMSE_log_var": rmse_log_var(true_var, pred_var),
        "QLIKE": qlike(true_var, pred_var),
        "MAE_var": mae(true_var, pred_var),
        "MAPE_var": mape(true_var, pred_var),
    }
