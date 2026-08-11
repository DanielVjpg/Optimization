"""The shared input object passed to every module.

A :class:`Dataset` is the single hand-off point between data loading and
analysis. Modules receive one and are not allowed to reach back to files or
uploads themselves — which is what keeps a new module (staffing, layout) from
needing its own ingestion code.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from core.business_profile import BusinessProfile


@dataclass
class ItemAttributes:
    """Catalog attributes for one item, with profile defaults already applied.

    Modules read this instead of poking at the raw items table, so a missing
    catalog column degrades to a documented default in exactly one place.
    """

    item: str
    unit: str = "unit"
    unit_cost: float = 1.0
    lead_time_days: float = 3.0
    shelf_life_days: float | None = None
    case_pack: float = 1.0
    min_order_qty: float = 0.0
    margin_per_unit: float = 0.0
    on_hand: float | None = None
    category: str = ""
    supplier: str = ""
    inferred_fields: tuple[str, ...] = ()
    """Names of fields that came from a default rather than the catalog."""

    @property
    def is_perishable(self) -> bool:
        return self.shelf_life_days is not None and self.shelf_life_days > 0


@dataclass
class Dataset:
    """Validated inputs for one business, ready for any module to consume."""

    profile: BusinessProfile
    sales: pd.DataFrame | None = None
    purchases: pd.DataFrame | None = None
    items: pd.DataFrame | None = None
    demand_source: str = "sales"
    """Either ``"sales"`` (true consumption) or ``"purchases_proxy"`` (demand
    inferred from what was ordered, which is lumpy — see warnings)."""
    warnings: list[str] = field(default_factory=list)

    # -- demand access --------------------------------------------------
    def demand_long(self) -> pd.DataFrame:
        """Gap-filled daily demand as a tidy frame of ``date, item, quantity``.

        Each item's calendar runs from its first observed movement to the last
        date in the dataset, with unobserved days filled as zero demand. Days
        before an item first appears are left out rather than zero-filled, so a
        recently introduced item is not penalised with a history of fake zeros.
        """
        frame = self.sales if self.demand_source == "sales" else self.purchases
        if frame is None or frame.empty:
            return pd.DataFrame(columns=["date", "item", "quantity"])

        daily = (
            frame.groupby(["item", "date"], as_index=False)["quantity"]
            .sum()
            .sort_values(["item", "date"])
        )
        # Same-day returns/waste adjustments can net negative; demand cannot.
        daily["quantity"] = daily["quantity"].clip(lower=0.0)

        end = daily["date"].max()
        filled: list[pd.DataFrame] = []
        for item, group in daily.groupby("item", sort=True):
            calendar = pd.date_range(group["date"].min(), end, freq="D")
            series = (
                group.set_index("date")["quantity"]
                .reindex(calendar, fill_value=0.0)
                .rename("quantity")
            )
            series.index.name = "date"
            filled.append(series.reset_index().assign(item=item))

        if not filled:
            return pd.DataFrame(columns=["date", "item", "quantity"])
        return pd.concat(filled, ignore_index=True)[["date", "item", "quantity"]]

    def demand_series(self, item: str) -> pd.Series:
        """Gap-filled daily demand for one item, indexed by date."""
        long = self.demand_long()
        rows = long[long["item"] == item]
        series = rows.set_index("date")["quantity"].astype(float)
        series.index = pd.DatetimeIndex(series.index)
        return series.asfreq("D", fill_value=0.0).rename(item)

    def item_names(self) -> list[str]:
        """Every item with demand history, alphabetically."""
        long = self.demand_long()
        if long.empty:
            return []
        return sorted(long["item"].unique().tolist())

    def date_range(self) -> tuple[pd.Timestamp, pd.Timestamp] | None:
        frames = [f for f in (self.sales, self.purchases) if f is not None and not f.empty]
        if not frames:
            return None
        starts = [f["date"].min() for f in frames]
        ends = [f["date"].max() for f in frames]
        return min(starts), max(ends)

    # -- catalog access -------------------------------------------------
    def attributes(self, item: str) -> ItemAttributes:
        """Catalog attributes for ``item``, filling gaps from the profile."""
        row: pd.Series | None = None
        if self.items is not None and not self.items.empty:
            matches = self.items[self.items["item"] == item]
            if not matches.empty:
                row = matches.iloc[0]

        inferred: list[str] = []

        def pick(column: str, default):
            if row is None or column not in row.index:
                inferred.append(column)
                return default
            value = row[column]
            if value is None or (isinstance(value, float) and math.isnan(value)):
                inferred.append(column)
                return default
            return value

        unit_cost = float(pick("unit_cost", self._observed_unit_cost(item)))
        margin = pick("margin_per_unit", None)
        if margin is None or (isinstance(margin, float) and math.isnan(margin)):
            sell_price = pick("sell_price", None)
            if sell_price is not None and not (
                isinstance(sell_price, float) and math.isnan(sell_price)
            ):
                margin = max(float(sell_price) - unit_cost, 0.0)
            else:
                margin = unit_cost * self.profile.stockout_margin_multiple

        shelf_life = pick("shelf_life_days", self.profile.default_shelf_life_days)
        shelf_life = None if shelf_life in (None, 0) else float(shelf_life)

        return ItemAttributes(
            item=item,
            unit=str(pick("unit", "unit")),
            unit_cost=unit_cost,
            lead_time_days=float(pick("lead_time_days", self.profile.default_lead_time_days)),
            shelf_life_days=shelf_life,
            case_pack=max(float(pick("case_pack", self.profile.default_case_pack)), 0.0) or 1.0,
            min_order_qty=float(pick("min_order_qty", self.profile.default_min_order_qty)),
            margin_per_unit=float(margin),
            on_hand=self._optional_float(pick("on_hand", None)),
            category=str(pick("category", "") or ""),
            supplier=str(pick("supplier", "") or ""),
            inferred_fields=tuple(dict.fromkeys(inferred)),
        )

    def _observed_unit_cost(self, item: str) -> float:
        """Fall back to the average unit cost actually paid, then to 1.0.

        A cost of 1.0 keeps the math well-defined; the resulting dollar figures
        are meaningless, so callers surface an inferred-cost warning.
        """
        if self.purchases is None or "unit_cost" not in getattr(self.purchases, "columns", []):
            return 1.0
        rows = self.purchases[self.purchases["item"] == item]
        costs = rows["unit_cost"].dropna() if not rows.empty else pd.Series(dtype=float)
        costs = costs[costs > 0]
        if costs.empty:
            return 1.0
        return float(costs.mean())

    @staticmethod
    def _optional_float(value) -> float | None:
        if value is None:
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return None if np.isnan(number) else number

    # -- introspection --------------------------------------------------
    def available_inputs(self) -> set[str]:
        available = set()
        for key in ("sales", "purchases", "items"):
            frame = getattr(self, key)
            if frame is not None and not frame.empty:
                available.add(key)
        return available

    def describe(self) -> dict[str, object]:
        span = self.date_range()
        return {
            "business": self.profile.name,
            "items": len(self.item_names()),
            "demand_source": self.demand_source,
            "history_start": None if span is None else span[0].date().isoformat(),
            "history_end": None if span is None else span[1].date().isoformat(),
            "history_days": None if span is None else (span[1] - span[0]).days + 1,
            "sales_rows": 0 if self.sales is None else len(self.sales),
            "purchase_rows": 0 if self.purchases is None else len(self.purchases),
        }
