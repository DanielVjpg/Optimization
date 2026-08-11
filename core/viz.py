"""Shared Altair chart builders.

Charts live in ``core`` when their *shape* is generic — an actual-versus-
predicted time series looks the same whether the series is milk consumption or
barista hours. Module-specific charts stay in the module.
"""

from __future__ import annotations

import altair as alt
import pandas as pd

# One palette for the whole product, so every module reads as the same tool.
ACTUAL_COLOR = "#1f4e79"
FITTED_COLOR = "#c26a1c"
FORECAST_COLOR = "#2e8b57"
BAND_COLOR = "#2e8b57"


def forecast_chart(
    history: pd.Series,
    forecast: pd.Series,
    backtest: pd.Series | None = None,
    lower: pd.Series | None = None,
    upper: pd.Series | None = None,
    *,
    title: str = "",
    y_title: str = "Quantity",
    band_label: str = "Service-level range",
) -> alt.LayerChart:
    """Actual history, backtested predictions and the forward forecast.

    The backtest series is the honest part of the picture: those points were
    predicted from data the model had not seen, so the gap between the orange
    and blue lines is the error the safety stock is sized against.
    """
    layers: list[alt.Chart] = []

    def to_frame(series: pd.Series, label: str) -> pd.DataFrame:
        frame = series.rename("quantity").reset_index()
        frame.columns = ["date", "quantity"]
        frame["series"] = label
        return frame

    if lower is not None and upper is not None and len(lower) and len(upper):
        band = pd.DataFrame(
            {
                "date": pd.DatetimeIndex(lower.index),
                "lower": lower.to_numpy(dtype=float),
                "upper": upper.to_numpy(dtype=float),
            }
        )
        layers.append(
            alt.Chart(band)
            .mark_area(opacity=0.16, color=BAND_COLOR)
            .encode(
                x=alt.X("date:T", title=None),
                y=alt.Y("lower:Q", title=y_title),
                y2="upper:Q",
                tooltip=[
                    alt.Tooltip("date:T", title="Date"),
                    alt.Tooltip("lower:Q", title=f"{band_label} low", format=",.1f"),
                    alt.Tooltip("upper:Q", title=f"{band_label} high", format=",.1f"),
                ],
            )
        )

    pieces = [(history, "Actual"), (backtest, "Backtest prediction"), (forecast, "Forecast")]
    frames = [to_frame(s, label) for s, label in pieces if s is not None and len(s)]
    if frames:
        combined = pd.concat(frames, ignore_index=True)
        layers.append(
            alt.Chart(combined)
            .mark_line(point=False, strokeWidth=2)
            .encode(
                x=alt.X("date:T", title=None),
                y=alt.Y("quantity:Q", title=y_title),
                color=alt.Color(
                    "series:N",
                    title=None,
                    scale=alt.Scale(
                        domain=["Actual", "Backtest prediction", "Forecast"],
                        range=[ACTUAL_COLOR, FITTED_COLOR, FORECAST_COLOR],
                    ),
                    legend=alt.Legend(orient="top"),
                ),
                strokeDash=alt.condition(
                    alt.datum.series == "Backtest prediction",
                    alt.value([5, 3]),
                    alt.value([0]),
                ),
                tooltip=[
                    alt.Tooltip("date:T", title="Date"),
                    alt.Tooltip("series:N", title="Series"),
                    alt.Tooltip("quantity:Q", title=y_title, format=",.1f"),
                ],
            )
        )

    chart = alt.layer(*layers).properties(height=340, title=title)
    return chart.interactive(bind_y=False)


def bar_chart(
    frame: pd.DataFrame,
    *,
    category: str,
    value: str,
    color: str | None = None,
    title: str = "",
    value_title: str | None = None,
    value_format: str = ",.0f",
) -> alt.Chart:
    """Horizontal bar chart, sorted by magnitude."""
    encodings = {
        "y": alt.Y(f"{category}:N", sort="-x", title=None),
        "x": alt.X(f"{value}:Q", title=value_title or value),
        "tooltip": [
            alt.Tooltip(f"{category}:N", title=category.replace("_", " ").title()),
            alt.Tooltip(f"{value}:Q", title=value_title or value, format=value_format),
        ],
    }
    if color:
        encodings["color"] = alt.Color(f"{color}:N", title=None, legend=alt.Legend(orient="top"))
    return alt.Chart(frame).mark_bar().encode(**encodings).properties(height=max(160, 26 * len(frame)), title=title)
