"""Find over- and under-ordered items and price the inefficiency.

Two comparisons drive everything here:

* **Ordered vs. used.** Over the history window, did more come in than went
  out? Persistent surplus is money sitting in a walk-in; persistent deficit is
  a stockout waiting to happen.
* **Order size vs. the economic order quantity.** Even an item that balances
  over a quarter can be bought in the wrong-sized lots — too often (wasted
  labour) or too big (wasted cash and, for perishables, waste).

Every dollar figure is an estimate built on stated assumptions, and each one
records which assumptions it rests on. A number a café owner cannot argue with
is worth more than a big number they cannot believe.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from core.business_profile import BusinessProfile
from core.dataset import Dataset, ItemAttributes
from modules.inventory.policy import ItemPolicy, total_annual_cost

OVER_ORDER_RATIO = 1.10
"""Purchased/used above this counts as over-ordering (10% headroom for the
normal lumpiness of case-pack buying)."""

UNDER_ORDER_RATIO = 0.92
"""Purchased/used below this counts as under-ordering."""

MATERIAL_VALUE = 20.0
"""Ignore imbalances worth less than this. Recommending action on a $3
discrepancy costs more attention than it returns."""

OVERSIZED_ORDER_RATIO = 1.5
"""Average order this many times the EOQ counts as an oversized lot."""


@dataclass
class ItemDiagnostic:
    """Ordering-behaviour findings for one item."""

    item: str
    unit: str
    status: str
    unit_cost: float
    window_days: int

    units_used: float
    units_purchased: float
    coverage_ratio: float
    net_units: float

    order_events: int
    avg_order_quantity: float
    recommended_order_quantity: float
    days_between_orders: float

    excess_units: float
    excess_value: float
    shortfall_units: float

    annual_excess_holding_cost: float
    annual_spoilage_cost: float
    annual_lot_sizing_cost: float
    annual_stockout_cost: float

    working_capital_tied_up: float
    findings: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)

    @property
    def annual_opportunity(self) -> float:
        """Total estimated annual cost of the current ordering behaviour."""
        return (
            self.annual_excess_holding_cost
            + self.annual_spoilage_cost
            + self.annual_lot_sizing_cost
            + self.annual_stockout_cost
        )


@dataclass
class SavingsSummary:
    """Portfolio-level totals, broken out by where the money comes from."""

    annual_excess_holding_cost: float = 0.0
    annual_spoilage_cost: float = 0.0
    annual_lot_sizing_cost: float = 0.0
    annual_stockout_cost: float = 0.0
    working_capital_tied_up: float = 0.0
    over_ordered_items: int = 0
    under_ordered_items: int = 0
    oversized_lot_items: int = 0
    balanced_items: int = 0
    window_days: int = 0

    @property
    def total_annual_savings(self) -> float:
        return (
            self.annual_excess_holding_cost
            + self.annual_spoilage_cost
            + self.annual_lot_sizing_cost
            + self.annual_stockout_cost
        )

    def breakdown(self) -> pd.DataFrame:
        rows = [
            ("Reduced spoilage / write-offs", self.annual_spoilage_cost,
             "Perishable stock bought beyond what shelf life allows to be used."),
            ("Lower carrying cost on excess stock", self.annual_excess_holding_cost,
             "Capital, space and shrink on inventory bought but not consumed."),
            ("Better order sizing (EOQ)", self.annual_lot_sizing_cost,
             "Ordering plus holding cost of current lot sizes vs. the economic order quantity."),
            ("Avoided stockouts", self.annual_stockout_cost,
             "Contribution margin projected to be lost if current under-buying continues."),
        ]
        return pd.DataFrame(rows, columns=["source", "annual_value", "basis"])


def diagnose_item(
    item: str,
    policy: ItemPolicy,
    attributes: ItemAttributes,
    profile: BusinessProfile,
    used_units: float,
    purchased_units: float | None,
    order_events: int,
    window_days: int,
) -> ItemDiagnostic:
    """Score one item's ordering behaviour and price the gap."""
    findings: list[str] = []
    assumptions: list[str] = []
    window_days = max(int(window_days), 1)
    annualize = profile.days_per_year / window_days
    unit_cost = attributes.unit_cost

    has_purchases = purchased_units is not None
    purchased = float(purchased_units or 0.0)
    used = float(max(used_units, 0.0))

    coverage = purchased / used if (has_purchases and used > 0) else float("nan")
    net_units = purchased - used if has_purchases else 0.0

    excess_units = max(net_units, 0.0) if has_purchases else 0.0
    shortfall_units = max(-net_units, 0.0) if has_purchases else 0.0
    excess_value = excess_units * unit_cost

    avg_order_qty = purchased / order_events if order_events > 0 else 0.0
    days_between = window_days / order_events if order_events > 0 else float("nan")

    # Replenishment is a sawtooth, so purchases and usage never balance
    # exactly over a finite window — where the last delivery happened to land
    # leaves a few percent either way. Only a persistent gap is a finding, so
    # nothing is priced until the imbalance clears the ratio thresholds.
    is_over = bool(has_purchases and coverage == coverage and coverage >= OVER_ORDER_RATIO)
    is_under = bool(has_purchases and coverage == coverage and coverage <= UNDER_ORDER_RATIO)

    # --- carrying cost of stock bought but never used -------------------
    annual_excess_holding = 0.0
    if is_over and excess_value >= MATERIAL_VALUE:
        annual_excess_holding = excess_value * profile.holding_cost_rate
        assumptions.append(
            f"Surplus of {excess_units:,.1f} {attributes.unit} "
            f"({profile.money(excess_value)}) is assumed to sit on hand for a "
            f"year at a {profile.holding_cost_rate:.0%} carrying rate."
        )

    # --- spoilage on perishable surplus ---------------------------------
    annual_spoilage = 0.0
    if attributes.is_perishable and is_over and excess_units > 0:
        consumable = policy.mean_daily_demand * float(attributes.shelf_life_days or 0.0)
        spoiled = max(excess_units - consumable, 0.0)
        if spoiled * unit_cost >= MATERIAL_VALUE:
            annual_spoilage = spoiled * unit_cost * annualize
            findings.append(
                f"About {spoiled:,.1f} {attributes.unit} of the surplus cannot be "
                f"used within the {attributes.shelf_life_days:g}-day shelf life "
                "and is written off."
            )
            assumptions.append(
                "Spoilage assumes surplus beyond one shelf-life of demand is "
                "discarded at full cost, and that the pattern repeats annually."
            )

    # --- lot sizing: current order size vs. EOQ -------------------------
    annual_lot_sizing = 0.0
    if avg_order_qty > 0 and policy.annual_demand > 0 and policy.holding_cost_per_unit > 0:
        current = total_annual_cost(
            avg_order_qty, policy.annual_demand, profile.order_fixed_cost,
            policy.holding_cost_per_unit,
        )
        recommended = total_annual_cost(
            policy.order_quantity, policy.annual_demand, profile.order_fixed_cost,
            policy.holding_cost_per_unit,
        )
        annual_lot_sizing = max(current - recommended, 0.0)
        if annual_lot_sizing < MATERIAL_VALUE:
            annual_lot_sizing = 0.0
        elif avg_order_qty > policy.order_quantity * OVERSIZED_ORDER_RATIO:
            findings.append(
                f"Typical order of {avg_order_qty:,.1f} {attributes.unit} is "
                f"{avg_order_qty / policy.order_quantity:.1f}x the economic order "
                f"quantity of {policy.order_quantity:,.1f} — order smaller, more often."
            )
        elif policy.order_quantity > avg_order_qty * OVERSIZED_ORDER_RATIO:
            findings.append(
                f"Typical order of {avg_order_qty:,.1f} {attributes.unit} is well "
                f"below the economic order quantity of {policy.order_quantity:,.1f} — "
                "consolidating orders would cut handling cost."
            )
        if annual_lot_sizing > 0:
            assumptions.append(
                f"Lot-sizing saving compares ordering+holding cost at the current "
                f"average order of {avg_order_qty:,.1f} against "
                f"{policy.order_quantity:,.1f}, at "
                f"{profile.money(profile.order_fixed_cost)} per order."
            )

    # --- projected stockout cost ----------------------------------------
    annual_stockout = 0.0
    if is_under and shortfall_units > 0 and used > 0:
        projected = shortfall_units * annualize
        lost = projected * profile.stockout_realization_rate
        annual_stockout = lost * attributes.margin_per_unit
        if annual_stockout < MATERIAL_VALUE:
            annual_stockout = 0.0
        else:
            # Lead with the imbalance: it is the finding, and any lot-sizing
            # note is secondary advice about how to correct it.
            findings.insert(
                0,
                f"Purchases covered only {coverage:.0%} of usage over the window; "
                f"stock is being drawn down by {shortfall_units:,.1f} "
                f"{attributes.unit}.",
            )
            assumptions.append(
                "Stockout cost assumes the current shortfall continues and "
                f"becomes unmet demand ({projected:,.0f} {attributes.unit}/yr), "
                f"that {profile.stockout_realization_rate:.0%} of it is lost "
                "outright rather than substituted, at "
                f"{profile.money(attributes.margin_per_unit)} contribution "
                f"margin per {attributes.unit}."
            )

    # --- status ---------------------------------------------------------
    if not has_purchases:
        status = "No purchase data"
    elif np.isnan(coverage):
        status = "No usage"
    elif is_under and annual_stockout > 0:
        status = "Under-ordered"
    elif is_over and excess_value >= MATERIAL_VALUE:
        status = "Over-ordered"
        findings.insert(
            0,
            f"Bought {coverage - 1:.0%} more than was used — "
            f"{profile.money(excess_value)} of stock accumulated.",
        )
    elif annual_lot_sizing > 0 and avg_order_qty > policy.order_quantity * OVERSIZED_ORDER_RATIO:
        status = "Oversized orders"
    elif annual_lot_sizing > 0:
        status = "Review order size"
    else:
        status = "Balanced"

    return ItemDiagnostic(
        item=item,
        unit=attributes.unit,
        status=status,
        unit_cost=unit_cost,
        window_days=window_days,
        units_used=used,
        units_purchased=purchased,
        coverage_ratio=coverage,
        net_units=net_units,
        order_events=order_events,
        avg_order_quantity=avg_order_qty,
        recommended_order_quantity=policy.order_quantity,
        days_between_orders=days_between,
        excess_units=excess_units,
        excess_value=excess_value,
        shortfall_units=shortfall_units,
        annual_excess_holding_cost=annual_excess_holding,
        annual_spoilage_cost=annual_spoilage,
        annual_lot_sizing_cost=annual_lot_sizing,
        annual_stockout_cost=annual_stockout,
        working_capital_tied_up=excess_value,
        findings=findings,
        assumptions=assumptions,
    )


def diagnose_all(
    dataset: Dataset,
    policies: dict[str, ItemPolicy],
) -> tuple[dict[str, ItemDiagnostic], SavingsSummary]:
    """Diagnose every item and roll the results into a savings summary."""
    profile = dataset.profile
    demand = dataset.demand_long()
    span = dataset.date_range()
    window_days = 1 if span is None else max((span[1] - span[0]).days + 1, 1)

    used_by_item = (
        demand.groupby("item")["quantity"].sum() if not demand.empty else pd.Series(dtype=float)
    )

    purchases = dataset.purchases
    have_purchases = (
        purchases is not None and not purchases.empty and dataset.demand_source == "sales"
    )
    if have_purchases:
        purchased_by_item = purchases.groupby("item")["quantity"].sum()
        events_by_item = purchases.groupby("item")["date"].nunique()
    else:
        purchased_by_item = pd.Series(dtype=float)
        events_by_item = pd.Series(dtype=int)

    diagnostics: dict[str, ItemDiagnostic] = {}
    summary = SavingsSummary(window_days=window_days)

    for item, policy in policies.items():
        attributes = dataset.attributes(item)
        diagnostic = diagnose_item(
            item=item,
            policy=policy,
            attributes=attributes,
            profile=profile,
            used_units=float(used_by_item.get(item, 0.0)),
            purchased_units=float(purchased_by_item.get(item, 0.0)) if have_purchases else None,
            order_events=int(events_by_item.get(item, 0)) if have_purchases else 0,
            window_days=window_days,
        )
        diagnostics[item] = diagnostic

        summary.annual_excess_holding_cost += diagnostic.annual_excess_holding_cost
        summary.annual_spoilage_cost += diagnostic.annual_spoilage_cost
        summary.annual_lot_sizing_cost += diagnostic.annual_lot_sizing_cost
        summary.annual_stockout_cost += diagnostic.annual_stockout_cost
        summary.working_capital_tied_up += diagnostic.working_capital_tied_up

        if diagnostic.status == "Over-ordered":
            summary.over_ordered_items += 1
        elif diagnostic.status == "Under-ordered":
            summary.under_ordered_items += 1
        elif diagnostic.status == "Oversized orders":
            summary.oversized_lot_items += 1
        elif diagnostic.status == "Balanced":
            summary.balanced_items += 1

    return diagnostics, summary
