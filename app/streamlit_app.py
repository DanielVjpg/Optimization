"""Optimization — Streamlit app shell.

The shell owns three things and nothing else: loading data, editing the
business profile, and choosing a module. It has no knowledge of what any module
computes — it asks the registry what exists, calls ``run``, and hands the result
back to the module to draw. Adding staffing or layout later requires no change
to this file.

Run from the project root::

    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import modules  # noqa: F401,E402  — importing registers every module
from core import registry  # noqa: E402
from core.business_profile import BusinessProfile  # noqa: E402
from core.data_loader import build_dataset, load_sample_dataset  # noqa: E402
from core.module_base import ModuleInputError  # noqa: E402
from core.schemas import ITEMS_SCHEMA, PURCHASES_SCHEMA, SALES_SCHEMA, SchemaError  # noqa: E402

st.set_page_config(page_title="Optimization", page_icon="📦", layout="wide")


# ---------------------------------------------------------------------------
def sidebar_profile() -> BusinessProfile:
    """Business-level assumptions. Shared by every module, so they live here."""
    st.sidebar.subheader("Business profile")
    name = st.sidebar.text_input("Business name", value="Sample Coffee Shop")

    with st.sidebar.expander("Cost & service assumptions", expanded=False):
        service_level = st.slider(
            "Target service level", min_value=0.80, max_value=0.99, value=0.95, step=0.01,
            format="%.2f",
            help="Probability of not stocking out during a lead time. Higher "
                 "means more safety stock and more cash on the shelf.",
        )
        holding_cost_rate = st.slider(
            "Annual holding cost rate", min_value=0.05, max_value=0.60, value=0.25, step=0.01,
            format="%.2f",
            help="Cost of holding $1 of inventory for a year: capital, space, "
                 "insurance, shrink. 20-30% is typical in food service.",
        )
        order_fixed_cost = st.number_input(
            "Cost per item line ordered ($)", min_value=0.0, max_value=100.0, value=4.0, step=1.0,
            help="Incremental staff time to count, receive and put away ONE "
                 "item — not the cost of a whole delivery. Items sharing a "
                 "truck share that cost.",
        )
        max_days_of_cover = st.number_input(
            "Maximum days of cover", min_value=3.0, max_value=120.0, value=30.0, step=5.0,
            help="Practical ceiling on stock held, whatever EOQ says — a "
                 "stand-in for shelf space and cash.",
        )
        shelf_life_utilization = st.slider(
            "Shelf life used per order", min_value=0.2, max_value=1.0, value=0.5, step=0.05,
            format="%.2f",
            help="An order may span this fraction of an item's shelf life. "
                 "0.5 means a 12-day milk is ordered six days at a time.",
        )
        stockout_realization = st.slider(
            "Share of unmet demand actually lost", min_value=0.0, max_value=1.0,
            value=0.5, step=0.05, format="%.2f",
            help="When an item runs out, some customers substitute and some "
                 "walk. Drives the stockout cost estimate.",
        )
        default_lead_time = st.number_input(
            "Default lead time (days)", min_value=0.0, max_value=30.0, value=3.0, step=1.0,
            help="Used only for items with no lead time in the catalog.",
        )

    return BusinessProfile(
        name=name,
        service_level=service_level,
        holding_cost_rate=holding_cost_rate,
        order_fixed_cost=order_fixed_cost,
        max_days_of_cover=max_days_of_cover,
        shelf_life_utilization=shelf_life_utilization,
        stockout_realization_rate=stockout_realization,
        default_lead_time_days=default_lead_time,
    )


def sidebar_data(profile: BusinessProfile):
    """Sample data or uploads. The only place files enter the app."""
    st.sidebar.subheader("Data")
    source = st.sidebar.radio(
        "Source",
        ["Sample coffee shop", "Upload my own"],
        help="The sample dataset is synthetic but realistic, with a few "
             "ordering problems deliberately baked in.",
    )

    if source == "Sample coffee shop":
        return load_sample_dataset(profile=profile)

    sales = st.sidebar.file_uploader("Sales / usage history (CSV)", type=["csv"])
    purchases = st.sidebar.file_uploader("Purchase history (CSV, optional)", type=["csv"])
    items = st.sidebar.file_uploader("Item catalog (CSV, optional)", type=["csv"])

    if sales is None and purchases is None:
        return None

    return build_dataset(profile=profile, sales=sales, purchases=purchases, items=items)


def sidebar_module():
    """Module picker, including the roadmap entries as disabled options."""
    st.sidebar.subheader("Module")
    active = registry.active_modules()
    planned = registry.planned_modules()

    labels = {module.display_name: module for module in active}
    choice = st.sidebar.radio("Analysis", list(labels), label_visibility="collapsed")
    selected = labels[choice]
    st.sidebar.caption(selected.description)

    if planned:
        st.sidebar.markdown("**Coming next**")
        for module in planned:
            st.sidebar.markdown(
                f"<span style='opacity:0.55'>◦ {module.display_name} — {module.description}</span>",
                unsafe_allow_html=True,
            )
    return selected


def schema_help() -> None:
    """Show the expected input format, since a real client will ask."""
    with st.expander("What data does this need?"):
        st.markdown(
            "Column names are matched loosely — `Date`, `order_date`, `Item Name`, "
            "`Qty` and similar variants are all recognised, so a POS or supplier "
            "export usually works unedited."
        )
        for schema, required in (
            (SALES_SCHEMA, "Required (unless purchase history is supplied)"),
            (PURCHASES_SCHEMA, "Optional — needed to detect over/under-ordering"),
            (ITEMS_SCHEMA, "Optional — improves cost accuracy"),
        ):
            st.markdown(f"**`{schema.key}.csv`** — {required}")
            st.markdown(schema.description)
            rows = [
                {
                    "column": column.name,
                    "type": column.dtype,
                    "required": "yes" if column.required else "no",
                    "also accepts": ", ".join(column.aliases[:5]) or "—",
                    "meaning": column.description,
                }
                for column in schema.columns
            ]
            st.dataframe(rows, hide_index=True, use_container_width=True)


# ---------------------------------------------------------------------------
def main() -> None:
    st.title("📦 Optimization")
    st.caption(
        "Operations analysis for small businesses — industrial engineering and "
        "operations research applied to the decisions a café actually makes."
    )

    profile = sidebar_profile()
    st.session_state["money"] = profile.money

    module = sidebar_module()
    options = module.render_options()

    try:
        dataset = sidebar_data(profile)
    except SchemaError as error:
        st.error(f"**Could not read the uploaded data.**\n\n{error}")
        schema_help()
        return
    except FileNotFoundError as error:
        st.error(str(error))
        return

    if dataset is None:
        st.info(
            "Upload a sales/usage history to begin, or switch to the sample "
            "coffee shop in the sidebar to see the tool working."
        )
        schema_help()
        return

    st.markdown(f"### {profile.name} — {module.display_name}")

    try:
        with st.spinner("Forecasting demand and sizing replenishment…"):
            result = module.run(dataset, **options)
    except ModuleInputError as error:
        st.error(str(error))
        schema_help()
        return

    module.render(result)

    with st.container():
        st.divider()
        schema_help()
        st.caption(
            "Every dollar figure is an estimate derived from the assumptions "
            "shown in the sidebar and in each item's detail panel. Change the "
            "assumptions to see how much the conclusion depends on them."
        )


# Streamlit executes this file top to bottom on every interaction.
main()
