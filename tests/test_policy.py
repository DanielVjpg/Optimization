"""Inventory policy math, checked against hand-computed values."""

from __future__ import annotations

import math

import pytest

from core.business_profile import BusinessProfile
from modules.inventory.policy import (
    apply_order_constraints,
    economic_order_quantity,
    reorder_point,
    safety_stock,
    total_annual_cost,
)


class TestEOQ:
    def test_matches_hand_calculation(self):
        # D=1000, S=50, H=5  ->  sqrt(2*1000*50/5) = sqrt(20000) = 141.42
        assert economic_order_quantity(1000, 50, 5) == pytest.approx(math.sqrt(20000))

    def test_sits_at_the_minimum_of_the_total_cost_curve(self):
        demand, order_cost, holding = 5000.0, 30.0, 4.0
        q_star = economic_order_quantity(demand, order_cost, holding)
        best = total_annual_cost(q_star, demand, order_cost, holding)
        for factor in (0.5, 0.8, 0.95, 1.05, 1.25, 2.0):
            assert total_annual_cost(q_star * factor, demand, order_cost, holding) >= best

    def test_ordering_and_holding_costs_are_equal_at_the_optimum(self):
        demand, order_cost, holding = 2400.0, 60.0, 3.0
        q = economic_order_quantity(demand, order_cost, holding)
        assert (demand / q) * order_cost == pytest.approx((q / 2) * holding)

    def test_degenerate_inputs_return_zero(self):
        assert economic_order_quantity(0, 50, 5) == 0.0
        assert economic_order_quantity(1000, 0, 5) == 0.0
        assert economic_order_quantity(1000, 50, 0) == 0.0


class TestSafetyStock:
    def test_reduces_to_z_sigma_root_l_when_lead_time_is_certain(self):
        assert safety_stock(1.645, 10.0, 4.0) == pytest.approx(1.645 * 10.0 * 2.0)

    def test_lead_time_variability_increases_the_buffer(self):
        certain = safety_stock(1.645, 10.0, 4.0, sigma_lead_time_days=0.0, mean_daily_demand=50)
        variable = safety_stock(1.645, 10.0, 4.0, sigma_lead_time_days=1.0, mean_daily_demand=50)
        assert variable > certain

    def test_scales_with_the_service_level(self):
        low = BusinessProfile(service_level=0.90).z
        high = BusinessProfile(service_level=0.99).z
        assert safety_stock(high, 5.0, 3.0) > safety_stock(low, 5.0, 3.0)

    def test_zero_error_means_zero_buffer(self):
        assert safety_stock(1.645, 0.0, 5.0) == 0.0

    def test_negative_lead_time_is_rejected(self):
        with pytest.raises(ValueError):
            safety_stock(1.645, 5.0, -1.0)


class TestReorderPoint:
    def test_is_lead_time_demand_plus_buffer(self):
        assert reorder_point(120.0, 30.0) == pytest.approx(150.0)

    def test_never_negative(self):
        assert reorder_point(-5.0, 0.0) == 0.0


class TestOrderConstraints:
    def test_unconstrained_eoq_passes_through(self):
        quantity, binding, notes = apply_order_constraints(
            100.0, mean_daily_demand=10.0, shelf_life_days=None, case_pack=1, min_order_qty=0
        )
        assert quantity == 100.0
        assert binding == "EOQ"
        assert notes == []

    def test_shelf_life_caps_the_order(self):
        # 10/day with a 5-day shelf life used fully can never justify >50 units.
        quantity, binding, _ = apply_order_constraints(
            400.0, mean_daily_demand=10.0, shelf_life_days=5, case_pack=1,
            min_order_qty=0, shelf_life_utilization=1.0,
        )
        assert quantity == pytest.approx(50.0)
        assert binding == "Shelf life"

    def test_shelf_life_utilization_tightens_the_cap(self):
        quantity, binding, _ = apply_order_constraints(
            400.0, mean_daily_demand=10.0, shelf_life_days=5, case_pack=1,
            min_order_qty=0, shelf_life_utilization=0.5,
        )
        assert quantity == pytest.approx(25.0)
        assert binding == "Shelf life"

    def test_max_days_of_cover_caps_the_order(self):
        quantity, binding, _ = apply_order_constraints(
            5000.0, mean_daily_demand=10.0, shelf_life_days=None, case_pack=1,
            min_order_qty=0, max_days_of_cover=30.0,
        )
        assert quantity == pytest.approx(300.0)
        assert binding == "Max days of cover"

    def test_shelf_life_beats_max_cover_when_it_is_tighter(self):
        quantity, binding, _ = apply_order_constraints(
            5000.0, mean_daily_demand=10.0, shelf_life_days=8, case_pack=1,
            min_order_qty=0, max_days_of_cover=30.0, shelf_life_utilization=0.5,
        )
        assert quantity == pytest.approx(40.0)
        assert binding == "Shelf life"

    def test_supplier_minimum_raises_the_order(self):
        quantity, binding, _ = apply_order_constraints(
            3.0, mean_daily_demand=1.0, shelf_life_days=None, case_pack=1, min_order_qty=12
        )
        assert quantity == 12.0
        assert binding == "Supplier minimum"

    def test_rounds_up_to_a_whole_case(self):
        quantity, binding, _ = apply_order_constraints(
            13.0, mean_daily_demand=5.0, shelf_life_days=None, case_pack=6, min_order_qty=0
        )
        assert quantity == 18.0
        assert binding == "Case pack"

    def test_trivial_case_rounding_is_not_reported_as_binding(self):
        """Rounding up always moves the number a little; only a material move
        should claim the case pack drove the decision."""
        quantity, binding, _ = apply_order_constraints(
            98.0, mean_daily_demand=5.0, shelf_life_days=None, case_pack=6, min_order_qty=0
        )
        assert quantity == 102.0
        assert binding == "EOQ"

    def test_flags_a_case_pack_larger_than_shelf_life_allows(self):
        # 2 units/day, 2-day shelf life at 50% utilisation -> 2 usable, case 24.
        quantity, binding, notes = apply_order_constraints(
            30.0, mean_daily_demand=2.0, shelf_life_days=2, case_pack=24, min_order_qty=0
        )
        assert quantity == 24.0
        assert "exceeds shelf life" in binding
        assert notes and "shelf life" in notes[0]

    def test_no_flag_when_the_case_only_slightly_exceeds_shelf_life(self):
        # Cap is 20; one case of 4 rounds to 20 exactly — no conflict.
        _, binding, notes = apply_order_constraints(
            500.0, mean_daily_demand=10.0, shelf_life_days=4, case_pack=4,
            min_order_qty=0, shelf_life_utilization=0.5,
        )
        assert binding == "Shelf life"
        assert notes == []

    def test_no_demand_falls_back_to_one_case(self):
        quantity, binding, _ = apply_order_constraints(
            0.0, mean_daily_demand=0.0, shelf_life_days=None, case_pack=6, min_order_qty=0
        )
        assert quantity == 6.0
        assert binding == "No demand"


class TestTotalCost:
    def test_matches_hand_calculation(self):
        # (1200/100)*40 + (100/2)*6 = 480 + 300 = 780
        assert total_annual_cost(100, 1200, 40, 6) == pytest.approx(780.0)

    def test_zero_quantity_is_not_a_division_error(self):
        assert total_annual_cost(0, 1200, 40, 6) == 0.0
