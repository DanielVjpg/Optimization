"""The synthetic purchase-history generator.

The generator is only useful if the ordering behaviours it seeds are actually
detectable by the diagnostics, and if the unseeded majority stays inside the
"balanced" band. Both are asserted here.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

_spec = importlib.util.spec_from_file_location(
    "generate_purchase_history", PROJECT_ROOT / "data" / "generate_purchase_history.py"
)
gph = importlib.util.module_from_spec(_spec)
sys.modules["generate_purchase_history"] = gph
_spec.loader.exec_module(gph)


def growing_sales(items: dict[str, float], days: int = 181, growth: float = 2.5) -> pd.DataFrame:
    """Daily sales that grow by ``growth`` over the window, like the real data."""
    dates = pd.date_range("2023-01-01", periods=days, freq="D")
    ramp = np.linspace(1.0, growth, days)
    rows = []
    for item, base in items.items():
        quantities = np.round(base * ramp).astype(int)
        rows.extend(
            {"date": d, "item": item, "quantity": float(q)} for d, q in zip(dates, quantities)
        )
    return pd.DataFrame(rows)


class TestCadence:
    @pytest.mark.parametrize(
        "mean_daily,expected", [(20.0, 7), (5.0, 7), (4.9, 14), (1.0, 14), (0.9, 28), (0.1, 28)]
    )
    def test_faster_movers_are_ordered_more_often(self, mean_daily, expected):
        assert gph.cadence_for(mean_daily) == expected


class TestPackRounding:
    def test_fast_movers_round_to_fives(self):
        assert gph.round_to_pack(47.3, mean_daily=10.0) == 45.0
        assert gph.round_to_pack(48.0, mean_daily=10.0) == 50.0

    def test_slow_movers_round_to_whole_units(self):
        assert gph.round_to_pack(2.4, mean_daily=0.5) == 2.0

    def test_never_orders_a_fraction_or_nothing(self):
        assert gph.round_to_pack(0.1, mean_daily=0.2) == 1.0
        assert gph.round_to_pack(0.0, mean_daily=0.0) == 0.0


class TestBaselineBehaviour:
    def test_unseeded_items_track_demand_closely(self):
        """The baseline must stay inside the diagnostics' balanced band, or
        the seeded findings drown in false positives."""
        sales = growing_sales({f"Item {i}": 8.0 for i in range(6)})
        _, summary = gph.build_purchases(sales, behaviours=())
        assert summary["coverage_ratio"].between(0.92, 1.10).all()

    def test_tracking_keeps_up_with_a_growing_item(self):
        """A trailing-demand baseline would under-buy everything on this data.
        Tracking must not."""
        sales = growing_sales({"Riser": 10.0}, growth=3.0)
        _, summary = gph.build_purchases(sales, behaviours=())
        assert summary["coverage_ratio"].iloc[0] > 0.92

    def test_output_has_the_columns_the_app_expects(self):
        sales = growing_sales({"Item A": 6.0})
        purchases, _ = gph.build_purchases(sales, behaviours=())
        assert list(purchases.columns) == ["date", "item", "quantity"]
        assert (purchases["quantity"] > 0).all()

    def test_generation_is_deterministic(self):
        sales = growing_sales({"Item A": 6.0, "Item B": 2.0})
        first, _ = gph.build_purchases(sales, behaviours=())
        second, _ = gph.build_purchases(sales, behaviours=())
        pd.testing.assert_frame_equal(first, second)


class TestSeededBehaviours:
    def _coverage(self, behaviour, items, **kwargs):
        sales = growing_sales(items, **kwargs)
        _, summary = gph.build_purchases(sales, behaviours=(behaviour,))
        return summary.set_index("item").loc[behaviour.item, "coverage_ratio"]

    def test_fixed_weekly_over_orders(self):
        behaviour = gph.Behaviour(
            item="Syrup", kind="fixed_weekly", label="OVER-ORDERED", reason="",
            cadence_days=7, fixed_weekly_qty=70,
        )
        # ~35 used per week on average against a standing order of 70.
        assert self._coverage(behaviour, {"Syrup": 4.0}) > 1.10

    def test_case_minimum_over_orders(self):
        behaviour = gph.Behaviour(
            item="Beans", kind="case_minimum", label="OVER-ORDERED", reason="",
            cadence_days=42, case_size=24,
        )
        assert self._coverage(behaviour, {"Beans": 0.4}) > 1.10

    def test_lagged_under_orders_a_growing_item(self):
        behaviour = gph.Behaviour(
            item="Riser", kind="lagged", label="UNDER-ORDERED", reason="",
            cadence_days=7, lag_days=42,
        )
        assert self._coverage(behaviour, {"Riser": 8.0}, growth=2.5) < 0.92

    def test_lagged_is_harmless_on_flat_demand(self):
        """Reaction lag only hurts when demand is moving — worth asserting so
        the mechanism is understood as lag, not as a blanket multiplier."""
        behaviour = gph.Behaviour(
            item="Flat", kind="lagged", label="UNDER-ORDERED", reason="",
            cadence_days=7, lag_days=42,
        )
        assert self._coverage(behaviour, {"Flat": 8.0}, growth=1.0) == pytest.approx(1.0, abs=0.08)

    def test_a_seeded_item_missing_from_sales_fails_loudly(self):
        behaviour = gph.Behaviour(item="Ghost", kind="tracking", label="", reason="")
        with pytest.raises(SystemExit):
            gph.build_purchases(growing_sales({"Real": 5.0}), behaviours=(behaviour,))


class TestEndToEnd:
    def test_seeded_items_are_detected_by_the_diagnostics(self, tmp_path):
        """The whole point: what the generator seeds, the module must find."""
        from core.business_profile import BusinessProfile
        from core.data_loader import build_dataset
        from modules.inventory.module import InventoryModule

        over = gph.Behaviour(
            item="Syrup", kind="fixed_weekly", label="OVER-ORDERED", reason="",
            cadence_days=7, fixed_weekly_qty=70,
        )
        under = gph.Behaviour(
            item="Riser", kind="lagged", label="UNDER-ORDERED", reason="",
            cadence_days=7, lag_days=42,
        )
        sales = growing_sales({"Syrup": 4.0, "Riser": 8.0, "Steady": 6.0}, growth=2.5)
        purchases, _ = gph.build_purchases(sales, behaviours=(over, under))

        items = pd.DataFrame(
            [
                {"item": name, "unit_cost": 3.0, "margin_per_unit": 5.0, "lead_time_days": 3}
                for name in ("Syrup", "Riser", "Steady")
            ]
        )
        dataset = build_dataset(
            profile=BusinessProfile(order_fixed_cost=0.5),
            sales=sales, purchases=purchases, items=items,
        )
        table = InventoryModule().run(dataset).tables["diagnostics"].set_index("item")

        assert table.loc["Syrup", "status"] == "Over-ordered"
        assert table.loc["Riser", "status"] == "Under-ordered"
        assert table.loc["Steady", "status"] == "Balanced"
