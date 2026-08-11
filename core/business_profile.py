"""The business being optimized.

A :class:`BusinessProfile` carries the operating and cost assumptions that are
true of the *business*, not of any one module. Modules read what they need and
ignore the rest, which is why staffing/layout parameters can live here before
those modules exist.

Every default is a documented, arguable assumption. When running this for a
real client, override them from a YAML file rather than editing this module.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import NormalDist
from typing import Any


@dataclass
class BusinessProfile:
    """Operating and cost assumptions for one business."""

    # --- identity -------------------------------------------------------
    name: str = "Sample Coffee Shop"
    business_type: str = "coffee_shop"
    currency_symbol: str = "$"

    # --- calendar -------------------------------------------------------
    # Demand series are built on a full daily calendar (closed days appear as
    # zero demand), so annualization uses calendar days, not open days.
    days_per_year: int = 365

    # --- inventory assumptions -----------------------------------------
    service_level: float = 0.95
    """Target cycle service level: probability of not stocking out during a
    replenishment lead time. Drives the safety-stock z multiplier."""

    holding_cost_rate: float = 0.25
    """Annual cost of holding one dollar of inventory, as a fraction of unit
    cost. Covers capital, storage, insurance and shrink. 20-30% is the usual
    range quoted for food service."""

    order_fixed_cost: float = 4.0
    """Incremental cost of adding *one item line* to an order, in currency
    units: counting it during the pre-order check, receiving it, putting it
    away. Roughly ten minutes of a staffer's time.

    This is deliberately *not* the cost of a whole delivery. A café orders
    thirty items from one distributor on one truck; charging each item the full
    delivery cost inflates every EOQ into months of stock. Handling that
    properly is the joint replenishment problem — see the v2 roadmap.
    """

    max_days_of_cover: float = 30.0
    """Practical ceiling on how much stock to hold of anything, whatever EOQ
    says. Backstop for the shared-order-cost issue above, and a stand-in for
    the shelf space and cash a small business actually has."""

    shelf_life_utilization: float = 0.5
    """Fraction of an item's shelf life an order may span. At 0.5, a 12-day
    milk is ordered at most six days at a time — nobody wants to serve the last
    gallon on its expiry date, and demand may dip after delivery."""

    stockout_realization_rate: float = 0.5
    """Fraction of unmet demand assumed to be lost outright rather than
    substituted. When the oat milk runs out, some customers take whole milk and
    some walk. Halving the claim keeps the stockout estimate defensible; raise
    it for items with no substitute."""

    default_lead_time_days: float = 3.0
    default_shelf_life_days: float | None = None
    default_case_pack: float = 1.0
    default_min_order_qty: float = 0.0

    stockout_margin_multiple: float = 2.0
    """Fallback only. When an item has no ``margin_per_unit`` or ``sell_price``
    in the catalog, the contribution margin lost per unit short is estimated as
    ``unit_cost * stockout_margin_multiple``. Supply real margins when you can:
    this number drives the under-ordering cost estimate."""

    # --- assumptions reserved for planned modules -----------------------
    # Declared here so the profile schema is stable when those modules land.
    # Nothing reads these yet.
    labor_rate_per_hour: float = 18.0  # planned: /modules/staffing
    avg_service_time_seconds: float = 90.0  # planned: /modules/staffing
    floor_area_sqft: float | None = None  # planned: /modules/layout

    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        if not 0.5 <= self.service_level < 0.9999:
            raise ValueError(
                f"service_level must be in [0.5, 0.9999), got {self.service_level}"
            )
        if self.holding_cost_rate <= 0:
            raise ValueError("holding_cost_rate must be positive")
        if self.order_fixed_cost < 0:
            raise ValueError("order_fixed_cost cannot be negative")
        if self.days_per_year <= 0:
            raise ValueError("days_per_year must be positive")
        if not 0 < self.shelf_life_utilization <= 1:
            raise ValueError("shelf_life_utilization must be in (0, 1]")
        if not 0 <= self.stockout_realization_rate <= 1:
            raise ValueError("stockout_realization_rate must be in [0, 1]")
        if self.max_days_of_cover <= 0:
            raise ValueError("max_days_of_cover must be positive")

    @property
    def z(self) -> float:
        """Safety factor for the target service level.

        The inverse standard-normal CDF. At the default 95% service level this
        is 1.645, i.e. hold 1.645 forecast-error standard deviations of buffer
        over the lead time.
        """
        return float(NormalDist().inv_cdf(self.service_level))

    def money(self, amount: float) -> str:
        """Format a currency amount for display."""
        return f"{self.currency_symbol}{amount:,.2f}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    # ------------------------------------------------------------------
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BusinessProfile":
        """Build a profile from a dict, routing unknown keys into ``extra``.

        Unknown keys are preserved rather than rejected so a client's YAML can
        carry parameters for modules that do not exist yet.
        """
        known = {f for f in cls.__dataclass_fields__ if f != "extra"}
        kwargs = {k: v for k, v in data.items() if k in known}
        extra = {k: v for k, v in data.items() if k not in known}
        profile = cls(**kwargs)
        profile.extra.update(extra)
        return profile

    @classmethod
    def from_yaml(cls, path: str | Path) -> "BusinessProfile":
        """Load a profile from a YAML file."""
        import yaml  # imported lazily: only needed when a profile file is used

        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        if not isinstance(data, dict):
            raise ValueError(f"{path} must contain a YAML mapping")
        return cls.from_dict(data)
