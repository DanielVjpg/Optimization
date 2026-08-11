"""Shared test fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.business_profile import BusinessProfile  # noqa: E402
from core.data_loader import build_dataset, load_sample_dataset  # noqa: E402


@pytest.fixture(scope="session")
def profile() -> BusinessProfile:
    return BusinessProfile(name="Test Cafe")


@pytest.fixture(scope="session")
def sample_dataset(profile):
    return load_sample_dataset(profile=profile)


@pytest.fixture
def seasonal_series() -> pd.Series:
    """90 days of clean weekly seasonality plus a mild trend and light noise.

    Deliberately easy: a model that cannot beat a naive benchmark here is
    broken, not merely unlucky.
    """
    rng = np.random.default_rng(7)
    dates = pd.date_range("2026-01-01", periods=90, freq="D")
    weekday = np.array([0.8, 0.85, 0.9, 1.0, 1.2, 1.4, 1.15])
    level = 100 + 0.3 * np.arange(90)
    values = level * weekday[dates.dayofweek.to_numpy()] * rng.normal(1.0, 0.04, 90)
    return pd.Series(np.clip(values, 0, None), index=dates, name="test_item")


@pytest.fixture
def tiny_dataset(profile):
    """A minimal hand-built dataset: 8 weeks, two items, known catalog."""
    dates = pd.date_range("2026-01-01", periods=56, freq="D")
    sales = pd.DataFrame(
        [
            {"date": d, "item": item, "quantity": qty}
            for item, qty in (("Widget", 10.0), ("Gadget", 4.0))
            for d in dates
        ]
    )
    purchases = pd.DataFrame(
        [
            {"date": d, "item": "Widget", "quantity": 70.0, "unit_cost": 2.0}
            for d in dates[::7]
        ]
        + [
            {"date": d, "item": "Gadget", "quantity": 28.0, "unit_cost": 5.0}
            for d in dates[::7]
        ]
    )
    items = pd.DataFrame(
        [
            {
                "item": "Widget", "unit": "each", "unit_cost": 2.0, "lead_time_days": 3,
                "shelf_life_days": "", "case_pack": 1, "min_order_qty": 0,
                "margin_per_unit": 4.0, "on_hand": 40,
            },
            {
                "item": "Gadget", "unit": "each", "unit_cost": 5.0, "lead_time_days": 2,
                "shelf_life_days": 10, "case_pack": 6, "min_order_qty": 6,
                "margin_per_unit": 9.0, "on_hand": 5,
            },
        ]
    )
    return build_dataset(profile=profile, sales=sales, purchases=purchases, items=items)
