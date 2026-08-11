"""Generate the synthetic coffee-shop dataset the app ships with.

Run from the project root::

    python data/generate_sample_data.py

The data is deterministic (fixed seed) so results, screenshots and tests are
reproducible.

Design intent — this is a demo dataset, so it is built to exercise the analysis
rather than to be bland:

* Real weekly seasonality: weekends run ~35% above Mondays.
* A mild upward trend, so the trend-following model has something to find.
* A holiday dip and occasional weather-driven slow days, so backtests face
  genuine noise.
* **Deliberately seeded inefficiencies**, so the tool finds something on first
  run rather than reporting a clean bill of health:
    - Vanilla Syrup is over-ordered by ~60% — capital tied up in surplus
    - Oat Milk is under-ordered by ~22% — stock drawn down, stockout exposure
    - Croissants are over-ordered by ~20% and have a 2-day shelf life, so the
      surplus genuinely spoils, and their case pack is larger than that shelf
      life supports — which makes the shelf-life constraint bind visibly
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUTPUT_DIR = Path(__file__).resolve().parent / "sample"
SEED = 20260811
HISTORY_DAYS = 120
END_DATE = pd.Timestamp("2026-08-10")

# Monday .. Sunday. A café's week: quiet start, Friday lift, weekend peak.
DAY_OF_WEEK_FACTORS = np.array([0.88, 0.92, 0.96, 1.02, 1.16, 1.36, 1.20])
BASE_DAILY_DRINKS = 420.0
DAILY_GROWTH = 0.0006  # ~2% a month

# item, unit, per-drink usage rate, unit cost, lead time, shelf life,
# case pack, min order qty, contribution margin per stock unit
ITEMS = [
    ("Whole Milk",        "gallon", 0.0420, 4.25,  2,   12,    4,    4, 62.00),
    ("Oat Milk",          "carton", 0.0180, 3.10,  3,   45,   12,   12, 21.00),
    ("Espresso Beans",    "lb",     0.0125, 9.50,  5,   60,    5,    5, 98.00),
    ("Drip Coffee Beans", "lb",     0.0090, 7.25,  5,   60,    5,    5, 74.00),
    ("12oz Cups",         "unit",   0.4200, 0.11,  7, None, 1000, 1000,  0.90),
    ("16oz Cups",         "unit",   0.3100, 0.13,  7, None, 1000, 1000,  1.10),
    ("Cup Lids",          "unit",   0.7000, 0.04,  7, None, 1000, 1000,  0.60),
    ("Vanilla Syrup",     "bottle", 0.0065, 6.80,  4,  120,    6,    6, 34.00),
    ("Caramel Syrup",     "bottle", 0.0048, 6.80,  4,  120,    6,    6, 34.00),
    ("Croissants",        "unit",   0.0850, 1.35,  1,    2,   36,   36,  2.75),
    ("Matcha Powder",     "lb",     0.0021, 26.00, 6,   90,    2,    2, 210.00),
]

# How often each item is ordered, in days.
ORDER_CYCLE_DAYS = {
    "Whole Milk": 3,
    "Oat Milk": 7,
    "Espresso Beans": 7,
    "Drip Coffee Beans": 7,
    "12oz Cups": 21,
    "16oz Cups": 21,
    "Cup Lids": 21,
    "Vanilla Syrup": 14,
    "Caramel Syrup": 14,
    "Croissants": 2,
    "Matcha Powder": 21,
}

# The seeded inefficiencies: multiplier applied to what should have been ordered.
ORDER_BIAS = {
    "Vanilla Syrup": 1.60,   # over-ordered — capital tied up in surplus stock
    "Oat Milk": 0.78,        # under-ordered — stock drawn down, stockout risk
    "Croissants": 1.20,      # over-ordered perishable — surplus genuinely spoils
}


def build_traffic(rng: np.random.Generator, dates: pd.DatetimeIndex) -> np.ndarray:
    """Daily drink volume: trend x weekday seasonality x shocks x noise."""
    days = np.arange(len(dates))
    trend = BASE_DAILY_DRINKS * (1.0 + DAILY_GROWTH) ** days
    seasonal = DAY_OF_WEEK_FACTORS[dates.dayofweek.to_numpy()]

    shock = np.ones(len(dates))
    # Independence Day week: locals leave town.
    holiday = np.asarray(
        (dates >= pd.Timestamp("2026-07-01")) & (dates <= pd.Timestamp("2026-07-06"))
    )
    shock[holiday] *= 0.72
    # Closed Independence Day itself.
    shock[np.asarray(dates == pd.Timestamp("2026-07-04"))] = 0.0
    # Scattered bad-weather days.
    weather = rng.random(len(dates)) < 0.06
    shock[weather] *= rng.uniform(0.70, 0.85, size=weather.sum())

    noise = rng.lognormal(mean=0.0, sigma=0.075, size=len(dates))
    return np.maximum(trend * seasonal * shock * noise, 0.0)


def build_sales(rng: np.random.Generator, dates: pd.DatetimeIndex, traffic: np.ndarray) -> pd.DataFrame:
    """Daily consumption per item, driven by drink volume plus item-level noise."""
    rows = []
    for name, unit, rate, *_ in ITEMS:
        jitter = rng.normal(1.0, 0.06, size=len(dates))
        usage = traffic * rate * np.clip(jitter, 0.6, 1.5)
        # Whole units for countable items; two decimals for bulk measures.
        usage = np.round(usage) if unit == "unit" else np.round(usage, 2)
        for date, quantity in zip(dates, usage):
            if quantity <= 0:
                continue
            rows.append({"date": date.date().isoformat(), "item": name, "quantity": float(quantity)})
    return pd.DataFrame(rows)


def build_purchases(rng: np.random.Generator, sales: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Replenishment history: periodic orders sized to cover the next cycle.

    An owner ordering by feel: cover the coming cycle based on roughly what the
    last one used, rounded up to whole cases, with a bias per item where an
    inefficiency has been seeded.
    """
    rows = []
    catalog = {row[0]: row for row in ITEMS}

    for name, group in sales.groupby("item"):
        _, unit, _, unit_cost, _lead, _shelf, case_pack, min_qty, _margin = catalog[name]
        cycle = ORDER_CYCLE_DAYS[name]
        bias = ORDER_BIAS.get(name, 1.0)

        daily = group.set_index(pd.to_datetime(group["date"]))["quantity"].sort_index()
        daily = daily.reindex(dates, fill_value=0.0)

        for start in range(0, len(dates), cycle):
            window = daily.iloc[start : start + cycle]
            if window.empty:
                continue
            # Order on the last day of the previous cycle, sized off what that
            # cycle actually used — the ordering pattern of a shop without a
            # forecast, and the behaviour this tool is meant to replace.
            reference = daily.iloc[max(start - cycle, 0) : start]
            expected = float(reference.sum()) if len(reference) else float(window.sum())
            if expected <= 0:
                continue

            quantity = expected * bias * float(rng.normal(1.0, 0.08))
            quantity = max(quantity, 0.0)
            if case_pack > 1:
                # Round to the nearest whole case, not up: always rounding up
                # would bake a systematic over-order into every item and mask
                # the inefficiencies this dataset is meant to demonstrate.
                quantity = max(round(quantity / case_pack), 1) * case_pack
            quantity = max(quantity, float(min_qty))
            if quantity <= 0:
                continue

            # Small supplier price drift so unit_cost is not a constant column.
            cost = round(unit_cost * float(rng.normal(1.0, 0.02)), 4)
            rows.append(
                {
                    "date": dates[start].date().isoformat(),
                    "item": name,
                    "quantity": float(round(quantity, 2)),
                    "unit_cost": cost,
                }
            )

    return pd.DataFrame(rows).sort_values(["date", "item"]).reset_index(drop=True)


def build_items(rng: np.random.Generator, sales: pd.DataFrame) -> pd.DataFrame:
    """Item catalog, including a plausible current on-hand position."""
    rows = []
    for name, unit, _rate, unit_cost, lead, shelf, case_pack, min_qty, margin in ITEMS:
        recent = sales[sales["item"] == name].tail(14)["quantity"]
        daily = float(recent.mean()) if not recent.empty else 0.0
        # Scatter on-hand around the lead-time requirement so the "order now"
        # column is a genuine mix rather than uniformly green or uniformly red.
        cover_days = lead * float(rng.uniform(0.6, 2.4))
        on_hand = round(daily * cover_days, 2)
        rows.append(
            {
                "item": name,
                "unit": unit,
                "unit_cost": unit_cost,
                "lead_time_days": lead,
                "shelf_life_days": "" if shelf is None else shelf,
                "case_pack": case_pack,
                "min_order_qty": min_qty,
                "margin_per_unit": margin,
                "on_hand": on_hand,
                "category": _category(name),
                "supplier": _supplier(name),
            }
        )
    return pd.DataFrame(rows)


def _category(name: str) -> str:
    if "Milk" in name:
        return "Dairy & alternatives"
    if "Beans" in name or "Matcha" in name:
        return "Coffee & tea"
    if "Syrup" in name:
        return "Syrups"
    if "Croissant" in name:
        return "Bakery"
    return "Packaging"


def _supplier(name: str) -> str:
    return {
        "Dairy & alternatives": "Valley Dairy Co.",
        "Coffee & tea": "Northside Roasters",
        "Syrups": "Restaurant Depot",
        "Bakery": "Corner Bakehouse",
        "Packaging": "Restaurant Depot",
    }[_category(name)]


def main() -> None:
    rng = np.random.default_rng(SEED)
    dates = pd.date_range(end=END_DATE, periods=HISTORY_DAYS, freq="D")

    traffic = build_traffic(rng, dates)
    sales = build_sales(rng, dates, traffic)
    purchases = build_purchases(rng, sales, dates)
    items = build_items(rng, sales)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sales.to_csv(OUTPUT_DIR / "coffee_shop_sales.csv", index=False)
    purchases.to_csv(OUTPUT_DIR / "coffee_shop_purchases.csv", index=False)
    items.to_csv(OUTPUT_DIR / "coffee_shop_items.csv", index=False)

    print(f"Wrote sample data to {OUTPUT_DIR}")
    print(f"  {len(sales):>5} usage rows across {sales['item'].nunique()} items")
    print(f"  {len(purchases):>5} purchase rows")
    print(f"  {dates[0].date()} → {dates[-1].date()} ({len(dates)} days)")
    print(f"  seeded: over-ordered {list(ORDER_BIAS)[0]}, under-ordered {list(ORDER_BIAS)[1]}")


if __name__ == "__main__":
    main()
