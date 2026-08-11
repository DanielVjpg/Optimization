"""Forecast-accuracy metrics shared by any module that predicts something.

Kept in ``core`` rather than in the inventory module because a staffing model
predicting labour hours needs exactly the same error measures.
"""

from __future__ import annotations

import numpy as np


def _clean(actual, predicted) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(actual, dtype=float).ravel()
    p = np.asarray(predicted, dtype=float).ravel()
    if a.shape != p.shape:
        raise ValueError(f"shape mismatch: actual {a.shape} vs predicted {p.shape}")
    mask = np.isfinite(a) & np.isfinite(p)
    return a[mask], p[mask]


def mae(actual, predicted) -> float:
    """Mean absolute error, in units of the series."""
    a, p = _clean(actual, predicted)
    return float("nan") if a.size == 0 else float(np.mean(np.abs(a - p)))


def rmse(actual, predicted) -> float:
    """Root mean squared error. Penalises the large misses that cause
    stockouts, and is the error measure safety stock is derived from."""
    a, p = _clean(actual, predicted)
    return float("nan") if a.size == 0 else float(np.sqrt(np.mean((a - p) ** 2)))


def mape(actual, predicted) -> float:
    """Mean absolute percentage error, ignoring zero-demand days.

    Zero-demand days are common for slow movers and would make MAPE infinite,
    so they are excluded — which is also why model selection ranks on MAE.
    """
    a, p = _clean(actual, predicted)
    nonzero = a != 0
    if not nonzero.any():
        return float("nan")
    return float(np.mean(np.abs((a[nonzero] - p[nonzero]) / a[nonzero])) * 100.0)


def bias(actual, predicted) -> float:
    """Mean forecast error (predicted - actual).

    Positive means the model systematically over-forecasts, which quietly
    inflates order quantities; worth surfacing next to MAE.
    """
    a, p = _clean(actual, predicted)
    return float("nan") if a.size == 0 else float(np.mean(p - a))


def summarize(actual, predicted) -> dict[str, float]:
    return {
        "mae": mae(actual, predicted),
        "rmse": rmse(actual, predicted),
        "mape": mape(actual, predicted),
        "bias": bias(actual, predicted),
    }
