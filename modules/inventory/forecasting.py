"""Daily demand forecasting.

Three models sit behind one interface, and a rolling-origin backtest picks the
best one *per item*. Item-level selection matters: a café's milk is a smooth,
trending, strongly weekday-seasonal series, while its matcha powder may be
near-flat with sporadic zeros. One model does not serve both, and picking by
measured out-of-sample error is more defensible than picking by taste.

Everything is written out in numpy/pandas rather than pulled from a modelling
library, so the smoothing recursions are inspectable.

Why multiplicative day-of-week factors are non-negotiable here: coffee-shop
demand swings 30-40% between a Monday and a Saturday. A plain moving average
would under-order every Saturday and over-order every Monday, which is exactly
the failure mode this tool exists to remove.

Roadmap note (v2): with a year or more of history, Prophet or SARIMA would add
annual seasonality and holiday effects, and exogenous regressors (weather, local
events, promotions) would beat anything purely autoregressive. See README.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from core import metrics

MIN_DAYS_FOR_SEASONALITY = 14
MIN_DAYS_FOR_SMOOTHING = 21
DEFAULT_HORIZON = 14


# ---------------------------------------------------------------------------
# Day-of-week seasonality
# ---------------------------------------------------------------------------
def day_of_week_factors(series: pd.Series, min_days: int = MIN_DAYS_FOR_SEASONALITY) -> np.ndarray:
    """Multiplicative day-of-week factors, indexed Monday=0 .. Sunday=6.

    Estimated classically: divide each observation by a centred 7-day moving
    average (which removes level and trend, leaving the weekly pattern), then
    take the median ratio per weekday. The median rather than the mean so one
    catering order or one snow day does not redefine "Tuesday".

    Returns all-ones when there is too little history to estimate a pattern,
    which makes every downstream model degrade gracefully to non-seasonal.
    """
    factors = np.ones(7, dtype=float)
    values = series.astype(float)
    if len(values) < min_days:
        return factors

    centred = values.rolling(window=7, center=True, min_periods=7).mean()
    ratio = values / centred.replace(0.0, np.nan)
    ratio = ratio.replace([np.inf, -np.inf], np.nan).dropna()
    if ratio.empty:
        return factors

    weekdays = pd.DatetimeIndex(ratio.index).dayofweek
    for day in range(7):
        sample = ratio.to_numpy()[weekdays == day]
        if sample.size >= 2:
            factors[day] = float(np.median(sample))

    # Clip before normalising so a single wild ratio cannot drag the mean.
    factors = np.clip(factors, 0.2, 5.0)
    mean = factors.mean()
    return factors if mean <= 0 else factors / mean


def _deseasonalize(series: pd.Series, factors: np.ndarray) -> np.ndarray:
    weekdays = pd.DatetimeIndex(series.index).dayofweek.to_numpy()
    return series.to_numpy(dtype=float) / factors[weekdays]


def _future_index(series: pd.Series, horizon: int) -> pd.DatetimeIndex:
    start = pd.Timestamp(series.index[-1]) + pd.Timedelta(days=1)
    return pd.date_range(start, periods=horizon, freq="D")


# ---------------------------------------------------------------------------
# Forecasters
# ---------------------------------------------------------------------------
class Forecaster(ABC):
    """Fit on a daily demand series, predict the next ``horizon`` days.

    Implement this to add a model. Nothing else in the module needs to change:
    add it to :data:`CANDIDATE_MODELS` and the backtest will consider it.
    """

    name: str = "forecaster"

    def __init__(self) -> None:
        self._series: pd.Series | None = None
        self.factors: np.ndarray = np.ones(7)

    @abstractmethod
    def fit(self, series: pd.Series) -> "Forecaster":
        ...

    @abstractmethod
    def predict(self, horizon: int) -> pd.Series:
        ...

    def _check_fitted(self) -> pd.Series:
        if self._series is None:
            raise RuntimeError(f"{self.name} must be fitted before predicting")
        return self._series


class SeasonalNaiveForecaster(Forecaster):
    """Tomorrow looks like the same weekday last week.

    The benchmark every other model must beat. Surprisingly hard to beat on
    short, stable series — if it wins for an item, that is a real result, not a
    failure of the fancier models.
    """

    name = "Seasonal naive"

    def fit(self, series: pd.Series) -> "SeasonalNaiveForecaster":
        self._series = series.astype(float)
        return self

    def predict(self, horizon: int) -> pd.Series:
        series = self._check_fitted()
        window = series.to_numpy()[-7:] if len(series) >= 7 else series.to_numpy()
        if window.size == 0:
            window = np.zeros(1)
        repeats = int(np.ceil(horizon / window.size))
        values = np.tile(window, repeats)[:horizon]
        return pd.Series(np.clip(values, 0.0, None), index=_future_index(series, horizon))


class MovingAverageForecaster(Forecaster):
    """Flat level from a trailing average, re-seasonalised by weekday.

    Robust and nearly assumption-free. It cannot follow a trend, so it loses to
    Holt on a growing item and beats it on a noisy flat one.
    """

    name = "Moving average + weekday"

    def __init__(self, window: int = 28) -> None:
        super().__init__()
        self.window = window
        self.level = 0.0

    def fit(self, series: pd.Series) -> "MovingAverageForecaster":
        self._series = series.astype(float)
        self.factors = day_of_week_factors(self._series)
        deseasonalized = _deseasonalize(self._series, self.factors)
        window = min(self.window, deseasonalized.size)
        self.level = float(np.mean(deseasonalized[-window:])) if window else 0.0
        return self

    def predict(self, horizon: int) -> pd.Series:
        series = self._check_fitted()
        index = _future_index(series, horizon)
        values = self.level * self.factors[index.dayofweek.to_numpy()]
        return pd.Series(np.clip(values, 0.0, None), index=index)


class HoltWeekdayForecaster(Forecaster):
    """Damped Holt (level + trend) on the deseasonalised series.

    The default model. Recursions, with ``x`` the deseasonalised demand:

        forecast_t = level + phi * trend
        level_t    = alpha * x_t + (1 - alpha) * forecast_t
        trend_t    = beta * (level_t - level_{t-1}) + (1 - beta) * phi * trend_{t-1}

    and h steps ahead: ``(level + trend * sum_{i=1..h} phi^i) * weekday_factor``.

    ``phi < 1`` damps the trend. Undamped Holt extrapolated four weeks out will
    happily predict a café sells 40% more milk than it did last month because
    of a fortnight of good weather; damping is what keeps a 28-day horizon
    from turning a mild slope into a large over-order.

    ``alpha`` and ``beta`` are grid-searched on in-sample one-step error rather
    than fixed, because a busy staple and a slow syrup want very different
    smoothing.
    """

    name = "Holt smoothing + weekday"

    ALPHA_GRID: Sequence[float] = (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6)
    BETA_GRID: Sequence[float] = (0.0, 0.02, 0.05, 0.1, 0.2)

    def __init__(self, phi: float = 0.95) -> None:
        super().__init__()
        self.phi = phi
        self.alpha = 0.2
        self.beta = 0.05
        self.level = 0.0
        self.trend = 0.0

    @staticmethod
    def _run(x: np.ndarray, alpha: float, beta: float, phi: float) -> tuple[float, float, np.ndarray]:
        n = x.size
        level = float(np.mean(x[:7])) if n >= 7 else float(x[0])
        trend = float((np.mean(x[7:14]) - np.mean(x[:7])) / 7.0) if n >= 14 else 0.0
        fitted = np.empty(n, dtype=float)
        for t in range(n):
            prediction = level + phi * trend
            fitted[t] = prediction
            new_level = alpha * x[t] + (1.0 - alpha) * prediction
            trend = beta * (new_level - level) + (1.0 - beta) * phi * trend
            level = new_level
        return level, trend, fitted

    def fit(self, series: pd.Series) -> "HoltWeekdayForecaster":
        self._series = series.astype(float)
        self.factors = day_of_week_factors(self._series)
        x = _deseasonalize(self._series, self.factors)

        # Ignore the first week when scoring: those fitted values are still
        # dominated by the initialisation, not by the parameters being chosen.
        burn_in = min(7, max(x.size - 1, 0))
        best = (np.inf, self.alpha, self.beta)
        for alpha in self.ALPHA_GRID:
            for beta in self.BETA_GRID:
                _, _, fitted = self._run(x, alpha, beta, self.phi)
                sse = float(np.sum((x[burn_in:] - fitted[burn_in:]) ** 2))
                if np.isfinite(sse) and sse < best[0]:
                    best = (sse, alpha, beta)

        _, self.alpha, self.beta = best
        self.level, self.trend, _ = self._run(x, self.alpha, self.beta, self.phi)
        return self

    def predict(self, horizon: int) -> pd.Series:
        series = self._check_fitted()
        index = _future_index(series, horizon)
        steps = np.arange(1, horizon + 1)
        # sum_{i=1..h} phi^i — the damped cumulative trend contribution.
        damping = np.array([np.sum(self.phi ** np.arange(1, h + 1)) for h in steps])
        deseasonalized = self.level + self.trend * damping
        values = deseasonalized * self.factors[index.dayofweek.to_numpy()]
        return pd.Series(np.clip(values, 0.0, None), index=index)


ModelFactory = Callable[[], Forecaster]

CANDIDATE_MODELS: tuple[ModelFactory, ...] = (
    SeasonalNaiveForecaster,
    MovingAverageForecaster,
    HoltWeekdayForecaster,
)


# ---------------------------------------------------------------------------
# Backtesting and model selection
# ---------------------------------------------------------------------------
@dataclass
class BacktestOutcome:
    model_name: str
    predictions: pd.Series
    errors: np.ndarray
    metrics: dict[str, float]
    folds: int


@dataclass
class ItemForecast:
    """Everything downstream inventory math needs for one item."""

    item: str
    model_name: str
    horizon_days: int
    history: pd.Series
    forecast: pd.Series
    backtest: pd.Series
    sigma_daily_error: float
    """Std. dev. of daily forecast error, measured out-of-sample. This — not
    the variability of demand itself — is what safety stock must cover: you
    only need buffer for the part of demand you failed to anticipate."""
    metrics: dict[str, float]
    comparison: pd.DataFrame
    observations: int
    warnings: list[str] = field(default_factory=list)

    @property
    def mean_daily_forecast(self) -> float:
        return float(self.forecast.mean()) if len(self.forecast) else 0.0

    @property
    def mean_daily_history(self) -> float:
        return float(self.history.mean()) if len(self.history) else 0.0

    def demand_over(self, days: float) -> float:
        """Total forecast demand over the next ``days`` days.

        Uses the forecast itself rather than an average, so an order placed
        into a weekend is sized for a weekend. Beyond the forecast horizon it
        extends at the mean forecast rate.
        """
        if days <= 0:
            return 0.0
        values = self.forecast.to_numpy(dtype=float)
        horizon = values.size
        covered = min(float(days), float(horizon))
        whole = int(np.floor(covered))
        total = float(values[:whole].sum()) if whole else 0.0
        fraction = covered - whole
        if fraction > 0 and whole < horizon:
            total += float(values[whole]) * fraction
        if days > horizon:
            total += self.mean_daily_forecast * (days - horizon)
        return total


def rolling_origin_backtest(
    series: pd.Series,
    factory: ModelFactory,
    horizon: int = 7,
    n_splits: int = 4,
    min_train: int = 28,
) -> BacktestOutcome:
    """Walk-forward evaluation: fit on the past, score on the unseen next block.

    Repeats at several cut points so accuracy is not a verdict on one lucky
    week. This is the only place model quality is judged — in-sample fit is
    reported nowhere, because it rewards overfitting.
    """
    model = factory()
    values = series.astype(float)
    n = len(values)

    origins: list[int] = []
    for k in range(n_splits, 0, -1):
        cut = n - k * horizon
        if cut >= min_train:
            origins.append(cut)

    if not origins:
        # Too short to hold anything out. Fall back to one-step in-sample
        # residuals and let the caller warn that accuracy is optimistic.
        model.fit(values)
        prediction = model.predict(1)
        errors = np.array([float(values.iloc[-1] - prediction.iloc[0])])
        return BacktestOutcome(
            model_name=model.name,
            predictions=pd.Series(dtype=float),
            errors=errors,
            metrics={"mae": float(np.abs(errors).mean()), "rmse": float(np.abs(errors).mean()),
                     "mape": float("nan"), "bias": float(-errors[0])},
            folds=0,
        )

    predicted: list[pd.Series] = []
    actuals: list[np.ndarray] = []
    for cut in origins:
        train = values.iloc[:cut]
        test = values.iloc[cut : cut + horizon]
        if test.empty:
            continue
        prediction = factory().fit(train).predict(len(test))
        prediction.index = test.index
        predicted.append(prediction)
        actuals.append(test.to_numpy(dtype=float))

    combined = pd.concat(predicted).groupby(level=0).mean().sort_index()
    actual = np.concatenate(actuals)
    prediction_values = np.concatenate([p.to_numpy(dtype=float) for p in predicted])

    return BacktestOutcome(
        model_name=model.name,
        predictions=combined,
        errors=actual - prediction_values,
        metrics=metrics.summarize(actual, prediction_values),
        folds=len(origins),
    )


def forecast_item(
    series: pd.Series,
    item: str | None = None,
    horizon: int = DEFAULT_HORIZON,
    candidates: Sequence[ModelFactory] = CANDIDATE_MODELS,
    backtest_horizon: int = 7,
    n_splits: int = 4,
) -> ItemForecast:
    """Select the best model for one item by backtest MAE, then forecast.

    MAE is the selection criterion because it is in stock units and is not
    dominated by a single outlier day; RMSE is retained for safety stock, where
    the large misses are precisely what matters.
    """
    name = item or (series.name if series.name is not None else "item")
    series = series.astype(float).sort_index()
    warnings: list[str] = []
    observations = len(series)

    if observations == 0:
        empty = pd.Series(dtype=float)
        index = pd.date_range(pd.Timestamp.today().normalize(), periods=horizon, freq="D")
        return ItemForecast(
            item=str(name), model_name="none", horizon_days=horizon,
            history=empty, forecast=pd.Series(np.zeros(horizon), index=index),
            backtest=empty, sigma_daily_error=0.0, metrics={},
            comparison=pd.DataFrame(), observations=0,
            warnings=["No demand history for this item."],
        )

    if observations < MIN_DAYS_FOR_SMOOTHING:
        warnings.append(
            f"Only {observations} days of history: falling back to the seasonal "
            "naive benchmark and skipping model selection. Treat the "
            "recommendation as provisional."
        )
        usable: Sequence[ModelFactory] = (SeasonalNaiveForecaster,)
    else:
        usable = candidates

    outcomes: list[BacktestOutcome] = []
    for factory in usable:
        try:
            outcomes.append(
                rolling_origin_backtest(
                    series, factory, horizon=backtest_horizon, n_splits=n_splits,
                    min_train=max(MIN_DAYS_FOR_SEASONALITY, backtest_horizon * 2),
                )
            )
        except Exception as exc:  # a bad series must not kill the whole run
            warnings.append(f"{factory().name} failed on this item and was skipped ({exc}).")

    if not outcomes:
        outcomes = [
            BacktestOutcome(
                "Seasonal naive", pd.Series(dtype=float), np.array([0.0]),
                {"mae": float("nan"), "rmse": float("nan"), "mape": float("nan"), "bias": float("nan")},
                0,
            )
        ]
        warnings.append("No model could be backtested; using the seasonal naive benchmark.")

    if all(o.folds == 0 for o in outcomes):
        warnings.append(
            "History is too short to hold out a validation window, so accuracy "
            "is measured in-sample and is optimistic."
        )

    def rank(outcome: BacktestOutcome) -> tuple[float, float]:
        mae = outcome.metrics.get("mae", float("inf"))
        rmse = outcome.metrics.get("rmse", float("inf"))
        return (
            mae if np.isfinite(mae) else float("inf"),
            rmse if np.isfinite(rmse) else float("inf"),
        )

    best = min(outcomes, key=rank)
    factory_by_name = {f().name: f for f in usable}
    chosen_factory = factory_by_name.get(best.model_name, SeasonalNaiveForecaster)

    model = chosen_factory().fit(series)
    forecast = model.predict(horizon)

    sigma = _error_sigma(best.errors, series)

    comparison = pd.DataFrame(
        [
            {
                "model": o.model_name,
                "mae": o.metrics.get("mae", float("nan")),
                "rmse": o.metrics.get("rmse", float("nan")),
                "mape_pct": o.metrics.get("mape", float("nan")),
                "bias": o.metrics.get("bias", float("nan")),
                "folds": o.folds,
                "selected": o.model_name == best.model_name,
            }
            for o in outcomes
        ]
    ).sort_values("mae", na_position="last")

    return ItemForecast(
        item=str(name),
        model_name=best.model_name,
        horizon_days=horizon,
        history=series,
        forecast=forecast,
        backtest=best.predictions,
        sigma_daily_error=sigma,
        metrics=best.metrics,
        comparison=comparison.reset_index(drop=True),
        observations=observations,
        warnings=warnings,
    )


def _error_sigma(errors: np.ndarray, series: pd.Series) -> float:
    """Standard deviation of daily forecast error, with sane fallbacks.

    RMSE of the backtest residuals is used directly (rather than their sample
    std) because any systematic bias in the model is also demand the buffer has
    to absorb. A degenerate zero would produce zero safety stock, so it falls
    back to a fraction of demand variability.
    """
    finite = errors[np.isfinite(errors)] if errors.size else errors
    sigma = float(np.sqrt(np.mean(finite**2))) if finite.size else 0.0
    if not np.isfinite(sigma) or sigma <= 0:
        fallback = float(series.std(ddof=0)) if len(series) > 1 else 0.0
        sigma = fallback * 0.5
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = max(float(series.mean()) * 0.1, 0.0) if len(series) else 0.0
    return sigma


def forecast_all(
    demand: pd.DataFrame,
    horizon: int = DEFAULT_HORIZON,
    candidates: Sequence[ModelFactory] = CANDIDATE_MODELS,
) -> dict[str, ItemForecast]:
    """Forecast every item in a tidy ``date, item, quantity`` frame."""
    results: dict[str, ItemForecast] = {}
    for item, group in demand.groupby("item", sort=True):
        series = group.set_index("date")["quantity"].astype(float)
        series.index = pd.DatetimeIndex(series.index)
        series = series.asfreq("D", fill_value=0.0)
        results[str(item)] = forecast_item(series, item=str(item), horizon=horizon, candidates=candidates)
    return results
