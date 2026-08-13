"""Simulate a plausible purchase history to sit alongside real sales data.

Sales data tells you what a shop *used*. Without a matching record of what it
*bought*, the inventory module can recommend a policy but cannot tell you
whether current ordering is wrong — which is the finding an owner actually
cares about. When a client has POS exports but no clean purchase history (the
common case: invoices live in a shoebox), this generates a defensible stand-in.

Usage::

    python data/generate_purchase_history.py data/prepared/hell_s_kitchen_sales.csv \
        --catalog data/prepared/hell_s_kitchen_items.csv

THIS OUTPUT IS SYNTHETIC. It is honest about ordering *behaviour* — cadence,
round-number habits, case minimums, reaction lag — but it is not this shop's
real purchasing. Anything shown from it is a demonstration of what the analysis
would find, not a finding. Label it as such in any client-facing material.

How the simulation works
------------------------
Each item gets an order cadence based on its volume (fast movers weekly, slow
movers monthly), then one of several ordering *behaviours*:

``tracking``      order roughly what the coming cycle will need. The competent
                  default: an owner who eyeballs their own business well.
``fixed_weekly``  order the same round number every week regardless of demand.
``case_minimum``  order in a supplier case larger than the shop needs.
``bulk``          buy a case of something slow-moving every few months.
``lagged``        size orders on demand from weeks ago, so a growing item is
                  chronically under-supplied.

The baseline deliberately *tracks* demand rather than trailing it. This dataset
grows ~2.5x over six months, and an owner ordering off trailing demand would
under-buy every single item — which would swamp the seeded findings and tell
you nothing. Lagged ordering is therefore applied only where it is the point.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_SEED = 20230101
DEFAULT_COGS_RATIO = 0.30


@dataclass(frozen=True)
class Behaviour:
    """A deliberately imperfect ordering habit seeded onto one item."""

    item: str
    kind: str
    label: str
    reason: str
    cadence_days: int | None = None
    fixed_weekly_qty: float | None = None
    case_size: float | None = None
    lag_days: int = 42
    params: dict = field(default_factory=dict)


# --- The seeded story ------------------------------------------------------
# Chosen against the real demand profile so each one is a mistake a coffee
# shop plausibly makes, not an arbitrary multiplier.
SEEDED_BEHAVIOURS: tuple[Behaviour, ...] = (
    Behaviour(
        item="Chocolate syrup",
        kind="fixed_weekly",
        label="OVER-ORDERED",
        reason=(
            "Standing weekly order of a round 50 bottles, set when the shop was "
            "at its busiest and never revisited. Average usage is ~34/week, so "
            "the shop buys about half again as much as it pours."
        ),
        cadence_days=7,
        fixed_weekly_qty=50,
    ),
    Behaviour(
        item="Organic Decaf Blend",
        kind="case_minimum",
        label="OVER-ORDERED",
        reason=(
            "The distributor's minimum is a 24-bag case, ordered every six "
            "weeks. The shop sells about 20 bags in that time. A high unit "
            "cost ($28 retail) makes the surplus expensive to sit on."
        ),
        cadence_days=42,
        case_size=24,
    ),
    Behaviour(
        item="I Need My Bean! Diner mug",
        kind="bulk",
        label="OVER-ORDERED",
        reason=(
            "Branded merchandise bought a dozen at a time to keep the retail "
            "shelf looking full. It sells about five a month, so each case is "
            "more than two months of demand and stock keeps building."
        ),
        cadence_days=56,
        case_size=12,
    ),
    Behaviour(
        item="Ethiopia Lg",
        kind="lagged",
        label="UNDER-ORDERED",
        reason=(
            "Demand grew 2.5x over the window — one of the fastest-growing "
            "high-volume drinks — but orders are still sized on what the shop "
            "was using six weeks earlier. The order never catches up, so stock "
            "is drawn down continuously."
        ),
        cadence_days=7,
        lag_days=42,
    ),
    Behaviour(
        item="Hazelnut syrup",
        kind="lagged",
        label="UNDER-ORDERED",
        reason=(
            "Fastest-growing syrup on the menu (2.8x). Same reaction lag as "
            "Ethiopia Lg, and a syrup that runs out mid-shift costs a sale "
            "outright — there is no substitute a customer asked for."
        ),
        cadence_days=14,
        lag_days=42,
    ),
)


def cadence_for(mean_daily: float) -> int:
    """Order cadence a small shop would realistically use for this volume."""
    if mean_daily >= 5:
        return 7  # fast mover: weekly with the main delivery
    if mean_daily >= 1:
        return 14  # steady: biweekly
    return 28  # slow mover: monthly, if that


def round_to_pack(quantity: float, mean_daily: float) -> float:
    """Round to a quantity a person would actually write on an order form."""
    if quantity <= 0:
        return 0.0
    if mean_daily >= 5:
        return float(max(round(quantity / 5.0) * 5, 5))
    return float(max(round(quantity), 1))


def simulate_item(
    series: pd.Series,
    rng: np.random.Generator,
    behaviour: Behaviour | None,
    noise: float,
) -> list[tuple[pd.Timestamp, float]]:
    """Generate (date, quantity) orders for one item across the window."""
    mean_daily = float(series.mean())
    kind = behaviour.kind if behaviour else "tracking"
    cadence = (behaviour.cadence_days if behaviour and behaviour.cadence_days
               else cadence_for(mean_daily))

    orders: list[tuple[pd.Timestamp, float]] = []
    values = series.to_numpy(dtype=float)
    dates = series.index
    n = len(values)

    for start in range(0, n, cadence):
        order_date = dates[start]
        upcoming = values[start : start + cadence]
        if upcoming.size == 0:
            continue
        cycle_weeks = len(upcoming) / 7.0

        if kind == "fixed_weekly":
            # Same number every week, whatever the shop actually used.
            quantity = float(behaviour.fixed_weekly_qty) * cycle_weeks

        elif kind in ("case_minimum", "bulk"):
            # Whole cases, sized to at least cover the cycle.
            case = float(behaviour.case_size)
            needed = float(upcoming.sum())
            quantity = max(math.ceil(needed / case), 1) * case

        elif kind == "lagged":
            # Sized on demand from `lag_days` ago: never catches a growing item.
            lag = behaviour.lag_days
            reference = values[max(start - lag - cadence, 0) : max(start - lag, 0)]
            if reference.size == 0:
                reference = upcoming  # no history yet at the start of the window
            per_day = reference.mean()
            quantity = per_day * len(upcoming) * float(rng.normal(1.0, noise))

        else:  # "tracking" — the competent baseline
            quantity = float(upcoming.sum()) * float(rng.normal(1.0, noise))

        quantity = round_to_pack(max(quantity, 0.0), mean_daily)
        if quantity > 0:
            orders.append((order_date, quantity))

    return orders


def build_purchases(
    sales: pd.DataFrame,
    *,
    seed: int = DEFAULT_SEED,
    noise: float = 0.07,
    behaviours: tuple[Behaviour, ...] = SEEDED_BEHAVIOURS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate purchases for every item in a tidy sales frame."""
    rng = np.random.default_rng(seed)
    by_item = {b.item: b for b in behaviours}

    missing = sorted(set(by_item) - set(sales["item"]))
    if missing:
        raise SystemExit(
            "These seeded items are not in the sales data: "
            + ", ".join(missing)
            + "\nEdit SEEDED_BEHAVIOURS to match your dataset."
        )

    wide = (
        sales.pivot_table(index="date", columns="item", values="quantity", aggfunc="sum")
        .asfreq("D")
        .fillna(0.0)
        .sort_index()
    )

    rows: list[dict] = []
    summary: list[dict] = []
    for item in sorted(wide.columns):
        series = wide[item]
        behaviour = by_item.get(item)
        orders = simulate_item(series, rng, behaviour, noise)
        for order_date, quantity in orders:
            rows.append(
                {"date": order_date.strftime("%Y-%m-%d"), "item": item, "quantity": quantity}
            )
        used = float(series.sum())
        bought = float(sum(q for _, q in orders))
        summary.append(
            {
                "item": item,
                "behaviour": behaviour.kind if behaviour else "tracking",
                "seeded": behaviour.label if behaviour else "",
                "units_used": round(used, 1),
                "units_purchased": round(bought, 1),
                "coverage_ratio": round(bought / used, 3) if used > 0 else float("nan"),
                "orders": len(orders),
                "mean_daily": round(float(series.mean()), 2),
            }
        )

    purchases = (
        pd.DataFrame(rows).sort_values(["date", "item"]).reset_index(drop=True)
        if rows
        else pd.DataFrame(columns=["date", "item", "quantity"])
    )
    return purchases, pd.DataFrame(summary)


def attach_unit_cost(
    purchases: pd.DataFrame, catalog_path: Path, cogs_ratio: float
) -> tuple[pd.DataFrame, int]:
    """Add a unit_cost column estimated from menu price.

    Sales data carries what the shop *charges*, never what it *pays*. Without a
    cost the module's dollar figures fall back to a unit cost of 1.0 and mean
    nothing, so cost is estimated as a flat fraction of menu price. It is one
    assumption, applied uniformly, and it is the first thing to replace with
    real invoice costs.
    """
    catalog = pd.read_csv(catalog_path)
    if "sell_price" not in catalog.columns:
        raise SystemExit(f"{catalog_path} has no sell_price column.")
    prices = catalog.set_index("item")["sell_price"].to_dict()

    costs = purchases["item"].map(prices)
    matched = int(costs.notna().sum())
    purchases = purchases.copy()
    purchases["unit_cost"] = (costs * cogs_ratio).round(2)
    return purchases, matched


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("sales", type=Path, help="Cleaned sales CSV (date, item, quantity)")
    parser.add_argument("--catalog", type=Path, default=None,
                        help="Item catalog with sell_price, used to estimate unit_cost")
    parser.add_argument("--cogs-ratio", type=float, default=DEFAULT_COGS_RATIO,
                        help=f"Unit cost as a fraction of menu price (default {DEFAULT_COGS_RATIO})")
    parser.add_argument("--no-cost", action="store_true", help="Emit date,item,quantity only")
    parser.add_argument("--noise", type=float, default=0.07,
                        help="Order-size noise for well-managed items (default 0.07)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    if not args.sales.exists():
        raise SystemExit(f"No such file: {args.sales}")
    if not 0 < args.cogs_ratio <= 1:
        raise SystemExit("--cogs-ratio must be in (0, 1]")

    sales = pd.read_csv(args.sales)
    for column in ("date", "item", "quantity"):
        if column not in sales.columns:
            raise SystemExit(
                f"{args.sales} is missing {column!r}. Run data/prepare_pos_export.py first."
            )
    sales["date"] = pd.to_datetime(sales["date"], errors="coerce")
    sales = sales.dropna(subset=["date", "item", "quantity"])

    purchases, summary = build_purchases(sales, seed=args.seed, noise=args.noise)

    matched = 0
    if args.catalog and not args.no_cost:
        purchases, matched = attach_unit_cost(purchases, args.catalog, args.cogs_ratio)

    output = args.out or args.sales.with_name(
        args.sales.stem.replace("_sales", "") + "_purchases.csv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    purchases.to_csv(output, index=False)

    # --- report ---------------------------------------------------------
    span = (sales["date"].min().date(), sales["date"].max().date())
    print("SYNTHETIC purchase history — simulated ordering behaviour, not real invoices.")
    print()
    print(f"Items      {summary['item'].nunique()}")
    print(f"Window     {span[0]} → {span[1]}")
    print(f"Orders     {len(purchases):,} lines")
    if matched:
        print(
            f"Unit cost  estimated at {args.cogs_ratio:.0%} of menu price "
            f"for {matched:,}/{len(purchases):,} lines"
        )
    elif args.no_cost or not args.catalog:
        print("Unit cost  omitted — dollar figures in the app will be indicative only")

    seeded = summary[summary["seeded"] != ""]
    baseline = summary[summary["seeded"] == ""]
    print()
    print("Seeded findings")
    for label in ("OVER-ORDERED", "UNDER-ORDERED"):
        for _, row in seeded[seeded["seeded"] == label].iterrows():
            print(
                f"  {label:<14} {row['item']:<28} bought/used {row['coverage_ratio']:.2f}  "
                f"({row['units_purchased']:,.0f} vs {row['units_used']:,.0f} units, "
                f"{row['behaviour']})"
            )
    print()
    print(
        f"Baseline   {len(baseline)} items ordered to track demand; "
        f"coverage {baseline['coverage_ratio'].min():.2f}–"
        f"{baseline['coverage_ratio'].max():.2f} "
        f"(median {baseline['coverage_ratio'].median():.2f})"
    )
    print(f"Output     {output}  ({output.stat().st_size / 1024:,.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
