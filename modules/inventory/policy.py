"""Reorder points, safety stock and economic order quantities.

The textbook formulas, with the two adjustments that decide whether the output
is usable in an actual café:

1. Lead-time demand comes from the *forecast*, not from a historical average.
   An order placed Thursday for Saturday delivery must cover a weekend.
2. EOQ is treated as a starting point, then constrained by shelf life, case
   packs and supplier minimums. Unconstrained EOQ routinely tells a coffee shop
   to buy six weeks of milk.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from core.business_profile import BusinessProfile
from core.dataset import ItemAttributes
from modules.inventory.forecasting import ItemForecast

CASE_PACK_BINDING_THRESHOLD = 1.10
"""Rounding to a whole case counts as the binding constraint only when it
raises the order by more than this factor."""


# ---------------------------------------------------------------------------
# Formulas
# ---------------------------------------------------------------------------
def safety_stock(z: float, sigma_daily_error: float, lead_time_days: float,
                 sigma_lead_time_days: float = 0.0, mean_daily_demand: float = 0.0) -> float:
    """Buffer stock covering demand uncertainty over the lead time.

        SS = z * sqrt(L * sigma_d^2 + d^2 * sigma_L^2)

    The first term is demand uncertainty over L days (daily errors assumed
    independent, so variances add); the second is uncertainty in the lead time
    itself, which drops out when the supplier is reliable and reduces to the
    familiar ``SS = z * sigma_d * sqrt(L)``.

    ``sigma_d`` here is *forecast error*, not demand variability: predictable
    variation (every Saturday is busy) is already in the reorder point and
    should not be paid for twice in buffer stock.
    """
    if lead_time_days < 0:
        raise ValueError("lead_time_days cannot be negative")
    demand_variance = lead_time_days * (sigma_daily_error**2)
    lead_time_variance = (mean_daily_demand**2) * (sigma_lead_time_days**2)
    return float(z * math.sqrt(max(demand_variance + lead_time_variance, 0.0)))


def reorder_point(lead_time_demand: float, safety_stock_units: float) -> float:
    """Stock level that should trigger a new order.

        ROP = expected demand during the lead time + safety stock
    """
    return float(max(lead_time_demand + safety_stock_units, 0.0))


def economic_order_quantity(annual_demand: float, order_cost: float, holding_cost_per_unit: float) -> float:
    """Order size minimising ordering plus holding cost.

        Q* = sqrt(2 * D * S / H)

    Balances the fixed cost of ordering (fewer, larger orders) against the cost
    of holding stock (more, smaller orders). The total-cost curve is famously
    flat near the optimum, which is why the recommendation stays sensible even
    when S and H are only roughly known.
    """
    if annual_demand <= 0 or holding_cost_per_unit <= 0 or order_cost <= 0:
        return 0.0
    return float(math.sqrt((2.0 * annual_demand * order_cost) / holding_cost_per_unit))


def total_annual_cost(order_quantity: float, annual_demand: float, order_cost: float,
                      holding_cost_per_unit: float) -> float:
    """Ordering + holding cost for a given order size.

        TC(Q) = (D/Q) * S + (Q/2) * H

    Purchase cost of the goods is excluded: it is the same under any policy, so
    including it would bury the difference the policy actually makes.
    """
    if order_quantity <= 0 or annual_demand <= 0:
        return 0.0
    return float((annual_demand / order_quantity) * order_cost
                 + (order_quantity / 2.0) * holding_cost_per_unit)


def apply_order_constraints(
    quantity: float,
    *,
    mean_daily_demand: float,
    shelf_life_days: float | None,
    case_pack: float,
    min_order_qty: float,
    max_days_of_cover: float = 30.0,
    shelf_life_utilization: float = 0.5,
) -> tuple[float, str, list[str]]:
    """Bring a raw EOQ into what the supplier and the walk-in will allow.

    Returns the constrained quantity, the binding constraint, and any conflicts
    worth showing the operator. Order matters: cap for spoilage and for
    practical cover first, then raise to the supplier minimum, then round up to
    a whole case.
    """
    notes: list[str] = []
    binding = "EOQ"
    result = max(float(quantity), 0.0)

    shelf_life_cap = None
    if shelf_life_days and shelf_life_days > 0 and mean_daily_demand > 0:
        # An order must be consumed within its usable life, with a margin: the
        # last unit should not be served on its expiry date.
        shelf_life_cap = mean_daily_demand * float(shelf_life_days) * shelf_life_utilization
        if result > shelf_life_cap:
            result = shelf_life_cap
            binding = "Shelf life"

    if max_days_of_cover > 0 and mean_daily_demand > 0:
        cover_cap = mean_daily_demand * float(max_days_of_cover)
        if result > cover_cap:
            result = cover_cap
            binding = "Max days of cover"

    if min_order_qty and result < min_order_qty:
        result = float(min_order_qty)
        binding = "Supplier minimum"

    case_pack = float(case_pack) if case_pack and case_pack > 0 else 1.0
    if case_pack > 1:
        rounded = math.ceil(result / case_pack) * case_pack
        # Rounding up to a whole case always changes the number a little.
        # Only call the case pack the *binding* constraint when it moved the
        # order materially — otherwise every item would report "Case pack" and
        # the column would say nothing.
        if binding == "EOQ" and result > 0 and rounded > result * CASE_PACK_BINDING_THRESHOLD:
            binding = "Case pack"
        result = float(rounded)

    if shelf_life_cap is not None and result > shelf_life_cap * CASE_PACK_BINDING_THRESHOLD:
        notes.append(
            f"The smallest orderable quantity ({result:g}) exceeds what can be "
            f"used within the usable window of the {shelf_life_days:g}-day "
            f"shelf life at current demand (~{shelf_life_cap:,.1f}). Expect "
            "predictable waste — worth renegotiating pack size or splitting "
            "deliveries with the supplier."
        )
        binding = "Case pack (exceeds shelf life)"

    if result <= 0:
        result = max(case_pack, float(min_order_qty), 0.0)
        binding = "No demand"

    return float(result), binding, notes


# ---------------------------------------------------------------------------
# Per-item policy
# ---------------------------------------------------------------------------
@dataclass
class ItemPolicy:
    """The recommended replenishment policy for one item."""

    item: str
    unit: str
    unit_cost: float
    lead_time_days: float
    service_level: float
    z: float

    mean_daily_demand: float
    lead_time_demand: float
    sigma_daily_error: float
    safety_stock: float
    reorder_point: float

    annual_demand: float
    holding_cost_per_unit: float
    order_cost: float
    eoq_raw: float
    order_quantity: float
    binding_constraint: str

    orders_per_year: float
    days_of_cover: float
    annual_ordering_cost: float
    annual_holding_cost: float
    annual_total_cost: float

    on_hand: float | None = None
    order_now: bool | None = None
    forecast_model: str = ""
    forecast_mae: float = float("nan")
    notes: list[str] = field(default_factory=list)

    @property
    def cycle_stock_value(self) -> float:
        return (self.order_quantity / 2.0) * self.unit_cost

    @property
    def safety_stock_value(self) -> float:
        return self.safety_stock * self.unit_cost


def build_policy(
    forecast: ItemForecast,
    attributes: ItemAttributes,
    profile: BusinessProfile,
    sigma_lead_time_days: float = 0.0,
) -> ItemPolicy:
    """Turn a forecast plus catalog attributes into a replenishment policy."""
    lead_time = max(float(attributes.lead_time_days), 0.0)
    mean_daily = forecast.mean_daily_forecast

    # Annualised from the forward forecast so a trending item is sized on where
    # it is heading, not on where it has been.
    annual_demand = mean_daily * profile.days_per_year
    holding_cost_per_unit = attributes.unit_cost * profile.holding_cost_rate

    lead_time_demand = forecast.demand_over(lead_time)
    buffer = safety_stock(
        z=profile.z,
        sigma_daily_error=forecast.sigma_daily_error,
        lead_time_days=lead_time,
        sigma_lead_time_days=sigma_lead_time_days,
        mean_daily_demand=mean_daily,
    )
    rop = reorder_point(lead_time_demand, buffer)

    eoq_raw = economic_order_quantity(annual_demand, profile.order_fixed_cost, holding_cost_per_unit)
    quantity, binding, notes = apply_order_constraints(
        eoq_raw,
        mean_daily_demand=mean_daily,
        shelf_life_days=attributes.shelf_life_days,
        case_pack=attributes.case_pack,
        min_order_qty=attributes.min_order_qty,
        max_days_of_cover=profile.max_days_of_cover,
        shelf_life_utilization=profile.shelf_life_utilization,
    )

    orders_per_year = annual_demand / quantity if quantity > 0 else 0.0
    days_of_cover = quantity / mean_daily if mean_daily > 0 else float("inf")
    ordering_cost = orders_per_year * profile.order_fixed_cost
    holding_cost = (quantity / 2.0 + buffer) * holding_cost_per_unit

    if attributes.inferred_fields:
        inferred = ", ".join(attributes.inferred_fields)
        notes.append(f"Using default values for: {inferred}.")

    on_hand = attributes.on_hand
    return ItemPolicy(
        item=forecast.item,
        unit=attributes.unit,
        unit_cost=attributes.unit_cost,
        lead_time_days=lead_time,
        service_level=profile.service_level,
        z=profile.z,
        mean_daily_demand=mean_daily,
        lead_time_demand=lead_time_demand,
        sigma_daily_error=forecast.sigma_daily_error,
        safety_stock=buffer,
        reorder_point=rop,
        annual_demand=annual_demand,
        holding_cost_per_unit=holding_cost_per_unit,
        order_cost=profile.order_fixed_cost,
        eoq_raw=eoq_raw,
        order_quantity=quantity,
        binding_constraint=binding,
        orders_per_year=orders_per_year,
        days_of_cover=days_of_cover,
        annual_ordering_cost=ordering_cost,
        annual_holding_cost=holding_cost,
        annual_total_cost=ordering_cost + holding_cost,
        on_hand=on_hand,
        order_now=None if on_hand is None else bool(on_hand <= rop),
        forecast_model=forecast.model_name,
        forecast_mae=float(forecast.metrics.get("mae", float("nan"))),
        notes=notes,
    )
