"""Forecasting models, backtesting and model selection."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core import metrics
from modules.inventory.forecasting import (
    HoltWeekdayForecaster,
    MovingAverageForecaster,
    SeasonalNaiveForecaster,
    day_of_week_factors,
    forecast_item,
    rolling_origin_backtest,
)


class TestDayOfWeekFactors:
    def test_recovers_a_known_weekly_pattern(self, seasonal_series):
        factors = day_of_week_factors(seasonal_series)
        # Saturday (5) is the peak, Monday (0) the trough in the fixture.
        assert factors.argmax() == 5
        assert factors.argmin() == 0

    def test_factors_average_to_one(self, seasonal_series):
        assert day_of_week_factors(seasonal_series).mean() == pytest.approx(1.0)

    def test_flat_demand_gives_flat_factors(self):
        dates = pd.date_range("2026-01-01", periods=60, freq="D")
        flat = pd.Series(np.full(60, 25.0), index=dates)
        assert day_of_week_factors(flat) == pytest.approx(np.ones(7))

    def test_short_history_degrades_to_no_seasonality(self):
        dates = pd.date_range("2026-01-01", periods=6, freq="D")
        series = pd.Series([1, 2, 3, 4, 5, 6], index=dates, dtype=float)
        assert day_of_week_factors(series) == pytest.approx(np.ones(7))

    def test_survives_zero_demand_days(self):
        dates = pd.date_range("2026-01-01", periods=60, freq="D")
        values = np.tile([0, 10, 10, 10, 20, 30, 20], 9)[:60].astype(float)
        factors = day_of_week_factors(pd.Series(values, index=dates))
        assert np.all(np.isfinite(factors))


class TestForecasters:
    @pytest.mark.parametrize(
        "factory", [SeasonalNaiveForecaster, MovingAverageForecaster, HoltWeekdayForecaster]
    )
    def test_predictions_are_the_right_shape_and_non_negative(self, factory, seasonal_series):
        prediction = factory().fit(seasonal_series).predict(14)
        assert len(prediction) == 14
        assert (prediction >= 0).all()
        assert prediction.index[0] == seasonal_series.index[-1] + pd.Timedelta(days=1)

    @pytest.mark.parametrize(
        "factory", [SeasonalNaiveForecaster, MovingAverageForecaster, HoltWeekdayForecaster]
    )
    def test_predicting_before_fitting_is_an_error(self, factory):
        with pytest.raises(RuntimeError):
            factory().predict(7)

    def test_seasonal_models_reproduce_the_weekly_shape(self, seasonal_series):
        prediction = HoltWeekdayForecaster().fit(seasonal_series).predict(14)
        weekday_means = prediction.groupby(prediction.index.dayofweek).mean()
        assert weekday_means.idxmax() == 5
        assert weekday_means.idxmin() == 0

    def test_holt_follows_a_trend_that_the_moving_average_misses(self):
        dates = pd.date_range("2026-01-01", periods=90, freq="D")
        trending = pd.Series(50 + 1.5 * np.arange(90), index=dates, dtype=float)
        holt = HoltWeekdayForecaster().fit(trending).predict(14)
        flat = MovingAverageForecaster().fit(trending).predict(14)
        assert holt.mean() > flat.mean()

    def test_damping_keeps_the_trend_from_running_away(self):
        dates = pd.date_range("2026-01-01", periods=90, freq="D")
        trending = pd.Series(50 + 1.5 * np.arange(90), index=dates, dtype=float)
        far = HoltWeekdayForecaster().fit(trending).predict(28)
        undamped_28_day_growth = 1.5 * 28
        assert far.iloc[-1] - trending.iloc[-1] < undamped_28_day_growth

    def test_negative_forecasts_are_clipped_to_zero(self):
        dates = pd.date_range("2026-01-01", periods=60, freq="D")
        collapsing = pd.Series(np.clip(60 - 1.2 * np.arange(60), 0, None), index=dates)
        assert (HoltWeekdayForecaster().fit(collapsing).predict(28) >= 0).all()


class TestBacktesting:
    def test_scores_on_held_out_data(self, seasonal_series):
        outcome = rolling_origin_backtest(seasonal_series, HoltWeekdayForecaster, horizon=7, n_splits=4)
        assert outcome.folds == 4
        assert outcome.errors.size == 28
        assert np.isfinite(outcome.metrics["mae"])

    def test_short_series_falls_back_instead_of_failing(self):
        dates = pd.date_range("2026-01-01", periods=10, freq="D")
        short = pd.Series(np.full(10, 5.0), index=dates)
        outcome = rolling_origin_backtest(short, SeasonalNaiveForecaster, horizon=7, n_splits=4)
        assert outcome.folds == 0
        assert outcome.errors.size == 1

    def test_smoothing_beats_the_naive_benchmark_on_a_clean_seasonal_series(self, seasonal_series):
        naive = rolling_origin_backtest(seasonal_series, SeasonalNaiveForecaster, horizon=7)
        holt = rolling_origin_backtest(seasonal_series, HoltWeekdayForecaster, horizon=7)
        assert holt.metrics["mae"] < naive.metrics["mae"]


class TestModelSelection:
    def test_selects_the_lowest_mae_model(self, seasonal_series):
        forecast = forecast_item(seasonal_series, item="test_item", horizon=14)
        comparison = forecast.comparison
        selected = comparison[comparison["selected"]].iloc[0]
        assert selected["mae"] == comparison["mae"].min()
        assert forecast.model_name == selected["model"]

    def test_produces_a_positive_error_sigma_for_safety_stock(self, seasonal_series):
        forecast = forecast_item(seasonal_series, horizon=14)
        assert forecast.sigma_daily_error > 0

    def test_forecast_error_is_smaller_than_raw_demand_variability(self, seasonal_series):
        """The point of modelling seasonality: buffer against what you missed,
        not against the whole weekly swing."""
        forecast = forecast_item(seasonal_series, horizon=14)
        assert forecast.sigma_daily_error < seasonal_series.std()

    def test_short_history_is_handled_and_warned_about(self):
        dates = pd.date_range("2026-01-01", periods=12, freq="D")
        series = pd.Series(np.full(12, 8.0), index=dates)
        forecast = forecast_item(series, item="new_item", horizon=14)
        assert len(forecast.forecast) == 14
        assert any("provisional" in w or "days of history" in w for w in forecast.warnings)

    def test_empty_history_returns_a_zero_forecast(self):
        forecast = forecast_item(pd.Series(dtype=float), item="ghost", horizon=7)
        assert len(forecast.forecast) == 7
        assert forecast.forecast.sum() == 0
        assert forecast.warnings

    def test_demand_over_a_window_uses_the_dated_forecast(self, seasonal_series):
        forecast = forecast_item(seasonal_series, horizon=14)
        three_days = forecast.demand_over(3)
        assert three_days == pytest.approx(forecast.forecast.iloc[:3].sum())

    def test_demand_over_a_fractional_window(self, seasonal_series):
        forecast = forecast_item(seasonal_series, horizon=14)
        expected = forecast.forecast.iloc[:2].sum() + 0.5 * forecast.forecast.iloc[2]
        assert forecast.demand_over(2.5) == pytest.approx(expected)

    def test_demand_beyond_the_horizon_extends_at_the_mean_rate(self, seasonal_series):
        forecast = forecast_item(seasonal_series, horizon=7)
        beyond = forecast.demand_over(14)
        assert beyond > forecast.forecast.sum()

    def test_zero_days_is_zero_demand(self, seasonal_series):
        assert forecast_item(seasonal_series, horizon=7).demand_over(0) == 0.0


class TestMetrics:
    def test_perfect_prediction_scores_zero(self):
        values = [1.0, 2.0, 3.0]
        assert metrics.mae(values, values) == 0.0
        assert metrics.rmse(values, values) == 0.0
        assert metrics.bias(values, values) == 0.0

    def test_bias_is_positive_when_over_forecasting(self):
        assert metrics.bias([10, 10], [12, 12]) == pytest.approx(2.0)

    def test_mape_ignores_zero_demand_days(self):
        assert metrics.mape([0.0, 10.0], [5.0, 11.0]) == pytest.approx(10.0)

    def test_mismatched_shapes_are_rejected(self):
        with pytest.raises(ValueError):
            metrics.mae([1, 2, 3], [1, 2])
