"""Streamlit rendering for the inventory module.

Kept apart from :mod:`module` so the analysis can run headless. A future
module ships its own ``ui.py`` and the app shell is untouched.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import streamlit as st

from core import viz
from core.module_base import ModuleResult
from modules.inventory.module import InventoryAnalysis

STATUS_STYLES = {
    "Over-ordered": ("🔴", "Buying more than is used"),
    "Under-ordered": ("🟠", "Buying less than is used"),
    "Oversized orders": ("🟡", "Right total, wrong lot size"),
    "Review order size": ("🟡", "Lot size worth revisiting"),
    "Balanced": ("🟢", "Ordering matches usage"),
}


def render_options() -> dict[str, Any]:
    """Module-specific sidebar controls."""
    st.sidebar.subheader("Forecast settings")
    horizon = st.sidebar.select_slider(
        "Forecast horizon",
        options=[7, 14, 21, 28],
        value=14,
        format_func=lambda days: f"{days // 7} week{'s' if days > 7 else ''}",
        help="How far ahead to project demand. Longer horizons are less "
             "accurate; 1-2 weeks matches most café ordering cycles.",
    )
    sigma_lead_time = st.sidebar.number_input(
        "Lead-time variability (days, std. dev.)",
        min_value=0.0, max_value=10.0, value=0.0, step=0.5,
        help="Set above zero if deliveries are unreliable. This adds a second "
             "term to the safety-stock formula and raises the buffer.",
    )
    return {"horizon_days": int(horizon), "sigma_lead_time_days": float(sigma_lead_time)}


def render(result: ModuleResult) -> None:
    analysis: InventoryAnalysis = result.payload
    if analysis is None or not analysis.policies:
        st.warning("No items could be analysed. Check the uploaded data.")
        return

    profile_money = st.session_state.get("money", lambda v: f"${v:,.2f}")

    _render_headline(result, analysis, profile_money)
    st.divider()
    _render_savings(result, analysis, profile_money)
    st.divider()
    _render_recommendations(result, analysis)
    st.divider()
    _render_item_detail(analysis, profile_money)
    st.divider()
    _render_diagnostics(result, analysis, profile_money)
    _render_methodology(analysis)


# ---------------------------------------------------------------------------
def _render_headline(result: ModuleResult, analysis: InventoryAnalysis, money) -> None:
    summary = result.summary
    savings = analysis.savings

    st.subheader("Summary")
    columns = st.columns(4)
    columns[0].metric(
        "Estimated annual opportunity",
        money(savings.total_annual_savings),
        help="Sum of the four savings sources below. An estimate built on "
             "stated assumptions, not a guarantee.",
    )
    columns[1].metric(
        "Cash tied up in surplus",
        money(savings.working_capital_tied_up),
        help="Value of stock bought but not used over the history window — "
             "a one-off release, not an annual saving.",
    )
    columns[2].metric(
        "Items over-ordered",
        savings.over_ordered_items,
        help="Purchases exceeded usage by more than 10%.",
    )
    columns[3].metric(
        "Items under-ordered",
        savings.under_ordered_items,
        help="Purchases covered less than 92% of usage.",
    )

    st.caption(
        f"{summary['items_analyzed']} items · "
        f"{summary['history'].get('history_days', 0)} days of history "
        f"({summary['history'].get('history_start')} → {summary['history'].get('history_end')}) · "
        f"{summary['horizon_days']}-day forecast · "
        f"{summary['service_level']:.0%} target service level"
    )

    if result.warnings:
        with st.expander(f"⚠️ Data notes ({len(result.warnings)})"):
            for warning in result.warnings:
                st.markdown(f"- {warning}")


def _render_savings(result: ModuleResult, analysis: InventoryAnalysis, money) -> None:
    st.subheader("Where the money is")
    breakdown = result.tables["savings_breakdown"]

    left, right = st.columns([3, 2])
    with left:
        chart_data = breakdown[breakdown["annual_value"] > 0]
        if chart_data.empty:
            st.info(
                "No material inefficiency found. Current ordering is close to "
                "the recommended policy — worth re-running as volumes change."
            )
        else:
            st.altair_chart(
                viz.bar_chart(
                    chart_data,
                    category="source",
                    value="annual_value",
                    value_title="Estimated annual value ($)",
                    value_format=",.0f",
                ),
                use_container_width=True,
            )
    with right:
        display = breakdown.copy()
        display["annual_value"] = display["annual_value"].map(money)
        st.dataframe(
            display.rename(columns={"source": "Source", "annual_value": "Annual", "basis": "Basis"}),
            hide_index=True,
            use_container_width=True,
        )

    st.caption(
        "These four figures are additive by construction but overlap in "
        "practice: carrying cost is charged on accumulated surplus, lot-sizing "
        "cost on average cycle stock. Treat the total as an order of magnitude, "
        "and the per-item detail below as the actionable part."
    )


def _render_recommendations(result: ModuleResult, analysis: InventoryAnalysis) -> None:
    st.subheader("Recommended ordering policy")
    table = result.tables["recommendations"]
    if table.empty:
        st.info("No recommendations to show.")
        return

    st.dataframe(
        table,
        hide_index=True,
        use_container_width=True,
        column_config={
            "item": st.column_config.TextColumn("Item"),
            "unit": st.column_config.TextColumn("Unit", width="small"),
            "avg_daily_demand": st.column_config.NumberColumn("Daily demand", format="%.2f"),
            "lead_time_days": st.column_config.NumberColumn("Lead time (d)", format="%.0f"),
            "lead_time_demand": st.column_config.NumberColumn("Lead-time demand", format="%.1f"),
            "safety_stock": st.column_config.NumberColumn(
                "Safety stock", format="%.1f",
                help="z × forecast-error σ × √lead time",
            ),
            "reorder_point": st.column_config.NumberColumn(
                "Reorder point", format="%.1f",
                help="Order when stock falls to this level.",
            ),
            "order_quantity": st.column_config.NumberColumn(
                "Order qty", format="%.1f",
                help="EOQ after shelf-life, case-pack and minimum-order limits.",
            ),
            "days_of_cover": st.column_config.NumberColumn("Days cover", format="%.1f"),
            "orders_per_year": st.column_config.NumberColumn("Orders/yr", format="%.1f"),
            "binding_constraint": st.column_config.TextColumn(
                "Set by", help="Which constraint decided the order quantity."
            ),
            "on_hand": st.column_config.NumberColumn("On hand", format="%.1f"),
            "action": st.column_config.TextColumn("Action", width="small"),
            "unit_cost": st.column_config.NumberColumn("Unit cost", format="$%.2f"),
            "order_value": st.column_config.NumberColumn("Order value", format="$%.2f"),
            "forecast_model": st.column_config.TextColumn("Model"),
        },
    )
    st.download_button(
        "Download recommendations (CSV)",
        table.to_csv(index=False).encode("utf-8"),
        file_name="inventory_recommendations.csv",
        mime="text/csv",
    )


def _render_item_detail(analysis: InventoryAnalysis, money) -> None:
    st.subheader("Forecast vs. actual")
    items = analysis.items()
    item = st.selectbox("Item", items, key="inventory_item_detail")
    forecast = analysis.forecasts[item]
    policy = analysis.policies[item]

    upper = lower = None
    if forecast.sigma_daily_error > 0:
        margin = policy.z * forecast.sigma_daily_error
        lower = (forecast.forecast - margin).clip(lower=0)
        upper = forecast.forecast + margin

    st.altair_chart(
        viz.forecast_chart(
            history=forecast.history,
            forecast=forecast.forecast,
            backtest=forecast.backtest,
            lower=lower,
            upper=upper,
            title=f"{item} — daily demand ({policy.unit})",
            y_title=f"Quantity ({policy.unit})",
        ),
        use_container_width=True,
    )
    st.caption(
        "Dashed orange is the backtest: predictions made from data the model "
        "had not seen. The shaded band is the "
        f"{policy.service_level:.0%} service-level range around the forecast — "
        "the uncertainty safety stock is sized to absorb."
    )

    columns = st.columns(4)
    columns[0].metric("Model selected", forecast.model_name)
    mae = forecast.metrics.get("mae", float("nan"))
    columns[1].metric(
        "Backtest MAE",
        "—" if not np.isfinite(mae) else f"{mae:,.2f} {policy.unit}",
        help="Average absolute error on held-out days.",
    )
    columns[2].metric("Reorder point", f"{policy.reorder_point:,.1f} {policy.unit}")
    columns[3].metric(
        "Order quantity",
        f"{policy.order_quantity:,.1f} {policy.unit}",
        help=f"Set by: {policy.binding_constraint}",
    )

    diagnostic = analysis.diagnostics.get(item)
    if diagnostic is not None and diagnostic.findings:
        icon, _ = STATUS_STYLES.get(diagnostic.status, ("•", ""))
        st.markdown(f"**{icon} {diagnostic.status}**")
        for finding in diagnostic.findings:
            st.markdown(f"- {finding}")

    with st.expander("Model comparison and assumptions for this item"):
        if not forecast.comparison.empty:
            st.markdown("**Backtest results — the model with the lowest MAE was selected**")
            st.dataframe(
                forecast.comparison.rename(
                    columns={
                        "model": "Model", "mae": "MAE", "rmse": "RMSE",
                        "mape_pct": "MAPE %", "bias": "Bias", "folds": "Folds",
                        "selected": "Selected",
                    }
                ),
                hide_index=True,
                use_container_width=True,
            )
        st.markdown(
            f"**Policy inputs** — lead time {policy.lead_time_days:g} days · "
            f"unit cost {money(policy.unit_cost)} · "
            f"holding cost {money(policy.holding_cost_per_unit)}/unit/yr · "
            f"order cost {money(policy.order_cost)} · "
            f"z = {policy.z:.3f} · forecast-error σ = {policy.sigma_daily_error:,.2f}/day"
        )
        for note in policy.notes:
            st.markdown(f"- {note}")
        if diagnostic is not None:
            for assumption in diagnostic.assumptions:
                st.markdown(f"- {assumption}")


def _render_diagnostics(result: ModuleResult, analysis: InventoryAnalysis, money) -> None:
    st.subheader("Over- and under-ordering")
    table = result.tables["diagnostics"]
    if table.empty:
        st.info("Purchase history is required to compare ordering against usage.")
        return

    flagged = table[table["status"].isin(["Over-ordered", "Under-ordered", "Oversized orders"])]
    if flagged.empty:
        st.success("No item is materially over- or under-ordered.")
    else:
        for _, row in flagged.iterrows():
            icon, meaning = STATUS_STYLES.get(row["status"], ("•", ""))
            diagnostic = analysis.diagnostics[row["item"]]
            with st.container(border=True):
                left, right = st.columns([3, 1])
                left.markdown(f"**{icon} {row['item']}** — {row['status'].lower()} · {meaning}")
                right.metric("Annual cost", money(row["annual_opportunity"]), label_visibility="collapsed")
                for finding in diagnostic.findings:
                    left.markdown(f"- {finding}")

    with st.expander("Full ordering comparison"):
        st.dataframe(
            table,
            hide_index=True,
            use_container_width=True,
            column_config={
                "item": st.column_config.TextColumn("Item"),
                "status": st.column_config.TextColumn("Status"),
                "units_used": st.column_config.NumberColumn("Used", format="%.1f"),
                "units_purchased": st.column_config.NumberColumn("Purchased", format="%.1f"),
                "coverage_ratio": st.column_config.NumberColumn(
                    "Purchased/used", format="%.2f",
                    help="1.00 means purchases exactly matched usage.",
                ),
                "avg_order_qty": st.column_config.NumberColumn("Avg order", format="%.1f"),
                "recommended_order_qty": st.column_config.NumberColumn("Rec. order", format="%.1f"),
                "excess_value": st.column_config.NumberColumn("Surplus $", format="$%.2f"),
                "annual_spoilage_cost": st.column_config.NumberColumn("Spoilage/yr", format="$%.2f"),
                "annual_excess_holding_cost": st.column_config.NumberColumn("Carrying/yr", format="$%.2f"),
                "annual_lot_sizing_cost": st.column_config.NumberColumn("Lot sizing/yr", format="$%.2f"),
                "annual_stockout_cost": st.column_config.NumberColumn("Stockout/yr", format="$%.2f"),
                "annual_opportunity": st.column_config.NumberColumn("Total/yr", format="$%.2f"),
            },
        )
        st.download_button(
            "Download diagnostics (CSV)",
            table.to_csv(index=False).encode("utf-8"),
            file_name="inventory_diagnostics.csv",
            mime="text/csv",
        )


def _render_methodology(analysis: InventoryAnalysis) -> None:
    with st.expander("Method — how these numbers are produced"):
        st.markdown(
            """
**Forecasting.** Three models compete for every item: a seasonal-naive
benchmark, a moving average with day-of-week factors, and damped Holt
smoothing with day-of-week factors. Each is scored by rolling-origin
backtesting — fit on the past, predict an unseen block, repeat at several
cut points — and the lowest-MAE model wins for that item. Weekly seasonality
is estimated as the median ratio of demand to a centred 7-day average.

**Safety stock.** `SS = z · √(L·σ² + d²·σ_L²)`, where `σ` is the *forecast
error* standard deviation measured out-of-sample, not the variability of
demand. Predictable weekly swings are already carried in the reorder point;
buffer is only for what the model failed to anticipate.

**Reorder point.** `ROP = forecast demand over the lead time + safety stock`.
Lead-time demand comes from the dated forecast, so an order that spans a
weekend is sized for a weekend.

**Order quantity.** `EOQ = √(2DS/H)`, then constrained: never more than one
shelf life of demand, never below the supplier minimum, rounded up to a whole
case. The constraint that bound is reported per item.

**Inefficiency cost.** Purchases are compared against usage over the history
window, and current average lot size against EOQ. Every figure annualises a
pattern observed over a finite window and is an estimate.
            """
        )
        st.caption(
            f"Forecast horizon {analysis.horizon_days} days · "
            f"{len(analysis.policies)} items analysed."
        )
