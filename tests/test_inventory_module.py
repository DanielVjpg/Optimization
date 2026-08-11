"""End-to-end behaviour of the inventory module, plus diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

from core import registry
from core.business_profile import BusinessProfile
from core.module_base import ModuleInputError
from modules.inventory.module import InventoryModule


@pytest.fixture(scope="module")
def result(request):
    dataset = request.getfixturevalue("sample_dataset")
    return InventoryModule().run(dataset, horizon_days=14)


class TestModuleContract:
    def test_the_inventory_module_is_registered(self):
        import modules  # noqa: F401  — import registers the modules

        assert any(m.key == "inventory" for m in registry.active_modules())

    def test_staffing_and_layout_are_declared_but_not_built(self):
        import modules  # noqa: F401

        planned = {m.key for m in registry.planned_modules()}
        assert {"staffing", "layout"} <= planned
        assert {m.key for m in registry.active_modules()} == {"inventory"}

    def test_a_dataset_without_demand_is_rejected(self, profile):
        from core.dataset import Dataset

        with pytest.raises(ModuleInputError):
            InventoryModule().run(Dataset(profile=profile))


class TestEndToEnd:
    def test_produces_every_expected_table(self, result):
        assert set(result.tables) == {
            "recommendations", "diagnostics", "savings_breakdown", "forecast_accuracy"
        }
        assert not result.tables["recommendations"].empty

    def test_covers_every_item_in_the_sample_data(self, result, sample_dataset):
        assert len(result.tables["recommendations"]) == len(sample_dataset.item_names())

    def test_recommendations_are_internally_consistent(self, result):
        table = result.tables["recommendations"]
        assert (table["reorder_point"] >= table["lead_time_demand"] - 1e-6).all()
        assert (table["safety_stock"] >= 0).all()
        assert (table["order_quantity"] > 0).all()
        assert (table["avg_daily_demand"] >= 0).all()

    def test_every_order_quantity_names_the_constraint_that_set_it(self, result):
        allowed = {"EOQ", "Shelf life", "Supplier minimum", "Case pack",
                   "Case pack (exceeds shelf life)", "No demand"}
        assert set(result.tables["recommendations"]["binding_constraint"]) <= allowed

    def test_case_packs_are_respected(self, result, sample_dataset):
        for _, row in result.tables["recommendations"].iterrows():
            case_pack = sample_dataset.attributes(row["item"]).case_pack
            if case_pack > 1:
                assert row["order_quantity"] % case_pack == pytest.approx(0, abs=0.05)

    def test_horizon_is_honoured(self, sample_dataset):
        analysis = InventoryModule().run(sample_dataset, horizon_days=28).payload
        assert all(len(f.forecast) == 28 for f in analysis.forecasts.values())

    def test_forecast_accuracy_is_reported_per_item(self, result):
        accuracy = result.tables["forecast_accuracy"]
        assert len(accuracy) == len(result.tables["recommendations"])
        assert accuracy["selected_model"].notna().all()

    def test_forecasts_are_reasonably_accurate_on_the_sample_data(self, result):
        """Sanity check on the whole pipeline: typical error should be a modest
        fraction of typical demand, not the same order as demand itself."""
        errors = result.tables["forecast_accuracy"]["mae_pct_of_mean"].dropna()
        assert errors.median() < 15.0


class TestServiceLevelBehaviour:
    def test_a_higher_service_level_buys_more_safety_stock(self, sample_dataset):
        from core.data_loader import load_sample_dataset

        low = InventoryModule().run(load_sample_dataset(BusinessProfile(service_level=0.85)))
        high = InventoryModule().run(load_sample_dataset(BusinessProfile(service_level=0.99)))
        assert (
            high.tables["recommendations"]["safety_stock"].sum()
            > low.tables["recommendations"]["safety_stock"].sum()
        )

    def test_a_higher_order_cost_pushes_toward_larger_orders(self):
        from core.data_loader import load_sample_dataset

        cheap = InventoryModule().run(load_sample_dataset(BusinessProfile(order_fixed_cost=5.0)))
        dear = InventoryModule().run(load_sample_dataset(BusinessProfile(order_fixed_cost=200.0)))
        assert (
            dear.tables["recommendations"]["orders_per_year"].sum()
            < cheap.tables["recommendations"]["orders_per_year"].sum()
        )


class TestDiagnostics:
    def test_finds_the_seeded_over_ordered_item(self, result):
        table = result.tables["diagnostics"].set_index("item")
        assert table.loc["Vanilla Syrup", "status"] == "Over-ordered"
        assert table.loc["Vanilla Syrup", "coverage_ratio"] > 1.3

    def test_finds_the_seeded_under_ordered_item(self, result):
        table = result.tables["diagnostics"].set_index("item")
        assert table.loc["Oat Milk", "status"] == "Under-ordered"
        assert table.loc["Oat Milk", "coverage_ratio"] < 0.9

    def test_does_not_flag_a_well_managed_item(self, result):
        table = result.tables["diagnostics"].set_index("item")
        assert table.loc["Caramel Syrup", "status"] not in {"Over-ordered", "Under-ordered"}

    def test_croissant_orders_are_capped_by_shelf_life(self, result):
        row = result.tables["recommendations"].set_index("item").loc["Croissants"]
        assert "Shelf life" in row["binding_constraint"] or "shelf life" in row["binding_constraint"]

    def test_prices_spoilage_on_an_over_ordered_perishable(self, result):
        table = result.tables["diagnostics"].set_index("item")
        assert table.loc["Croissants", "status"] == "Over-ordered"
        assert table.loc["Croissants", "annual_spoilage_cost"] > 0

    def test_normal_sawtooth_imbalance_is_not_priced_as_a_finding(self, result):
        """Purchases and usage never balance exactly over a finite window.
        A few percent either way is where the last delivery landed, not waste."""
        table = result.tables["diagnostics"].set_index("item")
        for item in ("12oz Cups", "Cup Lids", "Matcha Powder"):
            assert abs(table.loc[item, "coverage_ratio"] - 1.0) < 0.10
            assert table.loc[item, "annual_excess_holding_cost"] == 0.0
            assert table.loc[item, "annual_stockout_cost"] == 0.0

    def test_stockout_realization_rate_scales_the_stockout_estimate(self):
        from core.data_loader import load_sample_dataset

        half = InventoryModule().run(
            load_sample_dataset(BusinessProfile(stockout_realization_rate=0.5))
        )
        full = InventoryModule().run(
            load_sample_dataset(BusinessProfile(stockout_realization_rate=1.0))
        )
        assert full.tables["diagnostics"]["annual_stockout_cost"].sum() == pytest.approx(
            2 * half.tables["diagnostics"]["annual_stockout_cost"].sum()
        )

    def test_savings_are_positive_and_add_up(self, result):
        breakdown = result.tables["savings_breakdown"]
        total = result.summary["total_annual_savings"]
        assert total > 0
        assert breakdown["annual_value"].sum() == pytest.approx(total)

    def test_per_item_costs_sum_to_the_portfolio_total(self, result):
        per_item = result.tables["diagnostics"]["annual_opportunity"].sum()
        assert per_item == pytest.approx(result.summary["total_annual_savings"], rel=1e-3)

    def test_every_priced_finding_records_its_assumptions(self, result):
        analysis = result.payload
        for item, diagnostic in analysis.diagnostics.items():
            if diagnostic.annual_opportunity > 0:
                assert diagnostic.assumptions, f"{item} priced with no stated assumption"

    def test_a_balanced_shop_produces_no_material_findings(self, tiny_dataset):
        """The tiny fixture buys exactly what it uses, so nothing should be
        flagged as over- or under-ordered."""
        outcome = InventoryModule().run(tiny_dataset)
        statuses = set(outcome.tables["diagnostics"]["status"])
        assert "Over-ordered" not in statuses
        assert "Under-ordered" not in statuses

    def test_without_purchase_history_ordering_is_not_judged(self, profile):
        from core.data_loader import build_dataset
        import pandas as pd

        sales = pd.DataFrame(
            {
                "date": pd.date_range("2026-01-01", periods=60, freq="D"),
                "item": ["Milk"] * 60,
                "quantity": np.full(60, 12.0),
            }
        )
        dataset = build_dataset(profile=profile, sales=sales)
        outcome = InventoryModule().run(dataset)
        assert outcome.tables["diagnostics"]["status"].iloc[0] == "No purchase data"
        assert outcome.summary["total_annual_savings"] == 0.0
        assert any("purchase history" in w for w in outcome.warnings)
