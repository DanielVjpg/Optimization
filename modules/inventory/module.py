"""The inventory / demand-forecasting module.

Orchestration only — the analysis lives in :mod:`forecasting`, :mod:`policy`
and :mod:`diagnostics`, and the UI in :mod:`ui`. Keeping ``run`` free of both
math and Streamlit is what lets the same pipeline be driven from a test, a
notebook, or a consulting script that exports a client deliverable.

Pipeline:

    demand history -> forecast per item -> replenishment policy
                                        -> ordering diagnostics -> savings
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from core.dataset import Dataset
from core.module_base import ModuleResult, OptimizationModule
from modules.inventory import diagnostics as diag
from modules.inventory import forecasting, policy as policy_math
from modules.inventory.forecasting import DEFAULT_HORIZON, ItemForecast
from modules.inventory.policy import ItemPolicy


@dataclass
class InventoryAnalysis:
    """Rich per-item results, carried on :attr:`ModuleResult.payload`."""

    forecasts: dict[str, ItemForecast] = field(default_factory=dict)
    policies: dict[str, ItemPolicy] = field(default_factory=dict)
    diagnostics: dict[str, diag.ItemDiagnostic] = field(default_factory=dict)
    savings: diag.SavingsSummary = field(default_factory=diag.SavingsSummary)
    horizon_days: int = DEFAULT_HORIZON

    def items(self) -> list[str]:
        return sorted(self.policies)


class InventoryModule(OptimizationModule):
    """Forecast demand, size replenishment, and price ordering inefficiency."""

    key = "inventory"
    display_name = "Inventory & Demand Forecasting"
    description = (
        "Forecasts demand per item, recommends reorder points and order "
        "quantities from safety-stock and EOQ theory, and estimates the cost "
        "of current over- and under-ordering."
    )
    required_inputs = ("sales",)
    optional_inputs = ("purchases", "items")

    # ------------------------------------------------------------------
    def run(self, dataset: Dataset, **options: Any) -> ModuleResult:
        self.validate(dataset)

        horizon = int(options.get("horizon_days", DEFAULT_HORIZON))
        sigma_lead_time = float(options.get("sigma_lead_time_days", 0.0))
        selected_items = options.get("items")

        demand = dataset.demand_long()
        if selected_items:
            demand = demand[demand["item"].isin(selected_items)]

        forecasts = forecasting.forecast_all(demand, horizon=horizon)

        policies: dict[str, ItemPolicy] = {}
        for item, forecast in forecasts.items():
            policies[item] = policy_math.build_policy(
                forecast=forecast,
                attributes=dataset.attributes(item),
                profile=dataset.profile,
                sigma_lead_time_days=sigma_lead_time,
            )

        item_diagnostics, savings = diag.diagnose_all(dataset, policies)

        analysis = InventoryAnalysis(
            forecasts=forecasts,
            policies=policies,
            diagnostics=item_diagnostics,
            savings=savings,
            horizon_days=horizon,
        )

        warnings = list(dataset.warnings)
        warnings.extend(self._forecast_warnings(forecasts))
        if dataset.purchases is None:
            warnings.append(
                "No purchase history was provided, so over/under-ordering "
                "cannot be assessed — only the recommended policy is shown."
            )

        return ModuleResult(
            module_key=self.key,
            headline=(
                f"{dataset.profile.money(savings.total_annual_savings)} "
                "estimated annual opportunity"
            ),
            summary={
                "items_analyzed": len(policies),
                "horizon_days": horizon,
                "service_level": dataset.profile.service_level,
                "total_annual_savings": savings.total_annual_savings,
                "working_capital_tied_up": savings.working_capital_tied_up,
                "over_ordered_items": savings.over_ordered_items,
                "under_ordered_items": savings.under_ordered_items,
                "history": dataset.describe(),
            },
            tables={
                "recommendations": self.recommendations_table(analysis, dataset),
                "diagnostics": self.diagnostics_table(analysis),
                "savings_breakdown": savings.breakdown(),
                "forecast_accuracy": self.accuracy_table(analysis),
            },
            warnings=warnings,
            payload=analysis,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _forecast_warnings(forecasts: dict[str, ItemForecast]) -> list[str]:
        """Collapse per-item forecast warnings into one line per distinct issue."""
        collected: dict[str, list[str]] = {}
        for item, forecast in forecasts.items():
            for message in forecast.warnings:
                collected.setdefault(message, []).append(item)
        lines = []
        for message, items in collected.items():
            shown = ", ".join(sorted(items)[:4])
            more = "" if len(items) <= 4 else f" (+{len(items) - 4} more)"
            lines.append(f"{shown}{more}: {message}")
        return lines

    @staticmethod
    def recommendations_table(analysis: InventoryAnalysis, dataset: Dataset) -> pd.DataFrame:
        """One row per item: what to order, when, and how much."""
        rows = []
        for item in analysis.items():
            p = analysis.policies[item]
            rows.append(
                {
                    "item": item,
                    "unit": p.unit,
                    "avg_daily_demand": round(p.mean_daily_demand, 2),
                    "lead_time_days": p.lead_time_days,
                    "lead_time_demand": round(p.lead_time_demand, 1),
                    "safety_stock": round(p.safety_stock, 1),
                    "reorder_point": round(p.reorder_point, 1),
                    "order_quantity": round(p.order_quantity, 1),
                    "days_of_cover": (
                        None if not np.isfinite(p.days_of_cover) else round(p.days_of_cover, 1)
                    ),
                    "orders_per_year": round(p.orders_per_year, 1),
                    "binding_constraint": p.binding_constraint,
                    "on_hand": p.on_hand,
                    "action": (
                        "—" if p.order_now is None else ("Order now" if p.order_now else "OK")
                    ),
                    "unit_cost": round(p.unit_cost, 4),
                    "order_value": round(p.order_quantity * p.unit_cost, 2),
                    "forecast_model": p.forecast_model,
                }
            )
        return pd.DataFrame(rows)

    @staticmethod
    def diagnostics_table(analysis: InventoryAnalysis) -> pd.DataFrame:
        """One row per item: ordered vs. used, and what the gap costs."""
        rows = []
        for item in analysis.items():
            d = analysis.diagnostics.get(item)
            if d is None:
                continue
            rows.append(
                {
                    "item": item,
                    "status": d.status,
                    "units_used": round(d.units_used, 1),
                    "units_purchased": round(d.units_purchased, 1),
                    "coverage_ratio": (
                        None if np.isnan(d.coverage_ratio) else round(d.coverage_ratio, 2)
                    ),
                    "avg_order_qty": round(d.avg_order_quantity, 1),
                    "recommended_order_qty": round(d.recommended_order_quantity, 1),
                    "excess_value": round(d.excess_value, 2),
                    "annual_spoilage_cost": round(d.annual_spoilage_cost, 2),
                    "annual_excess_holding_cost": round(d.annual_excess_holding_cost, 2),
                    "annual_lot_sizing_cost": round(d.annual_lot_sizing_cost, 2),
                    "annual_stockout_cost": round(d.annual_stockout_cost, 2),
                    "annual_opportunity": round(d.annual_opportunity, 2),
                }
            )
        frame = pd.DataFrame(rows)
        if not frame.empty:
            frame = frame.sort_values("annual_opportunity", ascending=False).reset_index(drop=True)
        return frame

    @staticmethod
    def accuracy_table(analysis: InventoryAnalysis) -> pd.DataFrame:
        """Out-of-sample accuracy of the model chosen for each item."""
        rows = []
        for item in analysis.items():
            f = analysis.forecasts[item]
            mean_demand = f.mean_daily_history
            mae = f.metrics.get("mae", float("nan"))
            rows.append(
                {
                    "item": item,
                    "selected_model": f.model_name,
                    "mae": round(mae, 2) if np.isfinite(mae) else None,
                    "mae_pct_of_mean": (
                        round(100 * mae / mean_demand, 1)
                        if np.isfinite(mae) and mean_demand > 0
                        else None
                    ),
                    "rmse": round(f.metrics.get("rmse", float("nan")), 2),
                    "bias": round(f.metrics.get("bias", float("nan")), 2),
                    "sigma_daily_error": round(f.sigma_daily_error, 2),
                    "history_days": f.observations,
                }
            )
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    def render_options(self) -> dict[str, Any]:
        from modules.inventory import ui

        return ui.render_options()

    def render(self, result: ModuleResult) -> None:
        from modules.inventory import ui

        ui.render(result)
