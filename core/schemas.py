"""Canonical input tables shared by every module.

Real businesses export data with whatever column names their POS or supplier
portal happens to use. Rather than making the client reformat a spreadsheet,
each schema declares the canonical column plus the aliases seen in the wild;
:mod:`core.data_loader` maps an uploaded file onto the canonical form.

Three tables are defined:

``sales``      consumption events — what was actually used or sold
``purchases``  replenishment events — what was actually ordered in
``items``      the item catalog — costs and supplier constraints

Only a demand source is strictly required. Modules declare which tables they
need via :attr:`~core.module_base.OptimizationModule.required_inputs`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


class SchemaError(ValueError):
    """Raised when an input table cannot be mapped onto a canonical schema."""


def normalize_header(header: object) -> str:
    """Normalize a column header for alias matching.

    ``"Item Name "`` -> ``"item_name"``, ``"Qty."`` -> ``"qty"``.
    """
    text = str(header).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    dtype: str  # "date" | "text" | "number"
    required: bool = True
    aliases: tuple[str, ...] = ()
    description: str = ""

    def matches(self, normalized: str) -> bool:
        return normalized == self.name or normalized in self.aliases


@dataclass(frozen=True)
class TableSchema:
    key: str
    description: str
    columns: tuple[ColumnSpec, ...]

    @property
    def required_columns(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.columns if c.required)

    @property
    def column_names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.columns)

    def spec(self, name: str) -> ColumnSpec | None:
        for column in self.columns:
            if column.name == name:
                return column
        return None

    def resolve(self, headers: list[object]) -> dict[object, str]:
        """Map raw headers onto canonical column names.

        First match wins, so a file containing both ``quantity`` and ``qty``
        keeps the leftmost. Unmapped headers are simply dropped.
        """
        mapping: dict[object, str] = {}
        claimed: set[str] = set()
        for header in headers:
            normalized = normalize_header(header)
            for column in self.columns:
                if column.name in claimed:
                    continue
                if column.matches(normalized):
                    mapping[header] = column.name
                    claimed.add(column.name)
                    break
        return mapping


SALES_SCHEMA = TableSchema(
    key="sales",
    description=(
        "Consumption history: one row per item per day (or per transaction "
        "line). This is the demand signal the forecast is built from."
    ),
    columns=(
        ColumnSpec(
            "date",
            "date",
            aliases=(
                "order_date", "transaction_date", "business_date", "sale_date",
                "usage_date", "day", "datetime", "timestamp", "period",
            ),
            description="Date the item was used or sold.",
        ),
        ColumnSpec(
            "item",
            "text",
            aliases=(
                "item_name", "item_description", "product", "product_name",
                "sku", "ingredient", "material", "description", "name",
            ),
            description="Item identifier. Matched against the item catalog.",
        ),
        ColumnSpec(
            "quantity",
            "number",
            aliases=(
                "qty", "units", "quantity_used", "qty_used", "units_sold",
                "quantity_sold", "usage", "consumed", "count", "amount",
            ),
            description="Quantity consumed, in the item's stock unit.",
        ),
        ColumnSpec(
            "revenue",
            "number",
            required=False,
            aliases=("sales", "net_sales", "gross_sales", "total_sales", "revenue_usd"),
            description="Optional. Not used by the inventory module.",
        ),
    ),
)

PURCHASES_SCHEMA = TableSchema(
    key="purchases",
    description=(
        "Replenishment history: one row per item per delivery or purchase "
        "order line. Used to compare what was ordered against what was used."
    ),
    columns=(
        ColumnSpec(
            "date",
            "date",
            aliases=(
                "order_date", "po_date", "purchase_date", "received_date",
                "delivery_date", "invoice_date", "day", "datetime",
            ),
            description="Date the order was placed or received.",
        ),
        ColumnSpec(
            "item",
            "text",
            aliases=(
                "item_name", "item_description", "product", "product_name",
                "sku", "ingredient", "material", "description", "name",
            ),
        ),
        ColumnSpec(
            "quantity",
            "number",
            aliases=(
                "qty", "units", "quantity_ordered", "qty_ordered",
                "units_purchased", "ordered", "received", "amount",
            ),
            description="Quantity ordered, in the item's stock unit.",
        ),
        ColumnSpec(
            "unit_cost",
            "number",
            required=False,
            aliases=("cost", "price", "unit_price", "cost_per_unit", "unit_cost_usd"),
            description="Optional. Overrides the catalog cost when present.",
        ),
        ColumnSpec(
            "extended_cost",
            "number",
            required=False,
            aliases=("total_cost", "line_total", "extended_price", "line_cost"),
            description="Optional. Used to derive unit_cost when it is absent.",
        ),
    ),
)

ITEMS_SCHEMA = TableSchema(
    key="items",
    description=(
        "Item catalog: costs and supplier constraints. Every column except "
        "`item` is optional and falls back to the business profile default."
    ),
    columns=(
        ColumnSpec(
            "item",
            "text",
            aliases=("item_name", "product", "sku", "ingredient", "name"),
        ),
        ColumnSpec(
            "unit", "text", required=False,
            aliases=("uom", "stock_unit", "unit_of_measure", "measure"),
            description="Stock-keeping unit of measure, e.g. gallon, lb, case.",
        ),
        ColumnSpec(
            "unit_cost", "number", required=False,
            aliases=("cost", "price", "unit_price", "cost_per_unit"),
            description="Purchase cost of one stock unit.",
        ),
        ColumnSpec(
            "lead_time_days", "number", required=False,
            aliases=("lead_time", "leadtime", "lead_days", "supplier_lead_time"),
            description="Days between placing an order and it being usable.",
        ),
        ColumnSpec(
            "shelf_life_days", "number", required=False,
            aliases=("shelf_life", "expiry_days", "days_to_expiry", "life_days"),
            description="Usable life after delivery. Blank for non-perishables.",
        ),
        ColumnSpec(
            "case_pack", "number", required=False,
            aliases=("pack_size", "case_size", "units_per_case", "pack"),
            description="Units per purchasable case. Orders round up to this.",
        ),
        ColumnSpec(
            "min_order_qty", "number", required=False,
            aliases=("moq", "minimum_order", "min_order", "minimum_order_qty"),
        ),
        ColumnSpec(
            "margin_per_unit", "number", required=False,
            aliases=("contribution_margin", "margin", "gross_margin_per_unit"),
            description=(
                "Contribution margin lost per unit short. For an ingredient "
                "this is the margin of the drinks one stock unit produces."
            ),
        ),
        ColumnSpec(
            "sell_price", "number", required=False,
            aliases=("retail_price", "menu_price", "revenue_per_unit"),
            description="Alternative to margin_per_unit: margin = price - cost.",
        ),
        ColumnSpec(
            "on_hand", "number", required=False,
            aliases=("current_stock", "stock_on_hand", "inventory", "qty_on_hand"),
            description="Optional current stock, used to flag 'order now'.",
        ),
        ColumnSpec(
            "category", "text", required=False,
            aliases=("item_category", "group", "department", "type"),
        ),
        ColumnSpec(
            "supplier", "text", required=False,
            aliases=("vendor", "distributor", "supplier_name"),
        ),
    ),
)

SCHEMAS: dict[str, TableSchema] = {
    schema.key: schema for schema in (SALES_SCHEMA, PURCHASES_SCHEMA, ITEMS_SCHEMA)
}
