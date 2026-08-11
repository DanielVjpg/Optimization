"""Turn raw uploads into a validated :class:`~core.dataset.Dataset`.

This is the only place in the project that touches files or upload buffers.
Modules never read from disk, which is what lets a future module reuse the same
inputs without duplicating ingestion, cleaning or validation logic.
"""

from __future__ import annotations

from pathlib import Path
from typing import IO

import pandas as pd

from core.business_profile import BusinessProfile
from core.dataset import Dataset
from core.schemas import (
    ITEMS_SCHEMA,
    PURCHASES_SCHEMA,
    SALES_SCHEMA,
    SchemaError,
    TableSchema,
)

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "data" / "sample"

Source = str | Path | IO[bytes] | IO[str] | pd.DataFrame


def read_table(source: Source, schema: TableSchema) -> pd.DataFrame:
    """Read one table and map it onto ``schema``.

    Accepts a path, a file-like object (e.g. a Streamlit upload) or an
    already-loaded DataFrame. Raises :class:`SchemaError` when a required
    column cannot be resolved from the headers or their known aliases.
    """
    frame = source.copy() if isinstance(source, pd.DataFrame) else pd.read_csv(source)
    if frame.empty:
        raise SchemaError(f"The {schema.key} file is empty.")

    mapping = schema.resolve(list(frame.columns))
    missing = [c for c in schema.required_columns if c not in mapping.values()]
    if missing:
        raise SchemaError(
            f"The {schema.key} file is missing required column(s): "
            f"{', '.join(missing)}. Found headers: {', '.join(map(str, frame.columns))}. "
            f"Rename the columns, or use any recognised alias."
        )

    frame = frame[list(mapping.keys())].rename(columns=mapping)
    return _coerce(frame, schema)


def _coerce(frame: pd.DataFrame, schema: TableSchema) -> pd.DataFrame:
    """Cast columns to their declared types and drop unusable rows."""
    for name in frame.columns:
        spec = schema.spec(name)
        if spec is None:
            continue
        if spec.dtype == "date":
            frame[name] = pd.to_datetime(frame[name], errors="coerce").dt.normalize()
        elif spec.dtype == "number":
            frame[name] = pd.to_numeric(frame[name], errors="coerce")
        else:
            frame[name] = frame[name].astype("string").str.strip().str.replace(
                r"\s+", " ", regex=True
            )
            frame[name] = frame[name].replace({"": pd.NA})

    required = [c for c in schema.required_columns if c in frame.columns]
    cleaned = frame.dropna(subset=required)
    if cleaned.empty:
        raise SchemaError(
            f"Every row of the {schema.key} file was dropped during cleaning. "
            f"Check that {', '.join(required)} are populated and correctly formatted."
        )

    if "item" in cleaned.columns:
        cleaned["item"] = cleaned["item"].astype(str)
    return cleaned.reset_index(drop=True)


def build_dataset(
    profile: BusinessProfile | None = None,
    sales: Source | None = None,
    purchases: Source | None = None,
    items: Source | None = None,
) -> Dataset:
    """Assemble a :class:`Dataset` from whichever tables are available.

    At least one of ``sales`` or ``purchases`` must be supplied. When only
    purchases exist they stand in as the demand signal, with a warning: order
    history is lumpy (you buy four cases on Monday and nothing for a week), so
    it overstates day-to-day variability and therefore safety stock.
    """
    profile = profile or BusinessProfile()
    warnings: list[str] = []

    sales_df = read_table(sales, SALES_SCHEMA) if sales is not None else None
    purchases_df = read_table(purchases, PURCHASES_SCHEMA) if purchases is not None else None
    items_df = read_table(items, ITEMS_SCHEMA) if items is not None else None

    if sales_df is None and purchases_df is None:
        raise SchemaError(
            "Provide sales/usage history, purchase history, or both. "
            "Neither was supplied."
        )

    if purchases_df is not None:
        purchases_df = _derive_unit_cost(purchases_df)

    if sales_df is not None:
        demand_source = "sales"
    else:
        demand_source = "purchases_proxy"
        warnings.append(
            "No sales/usage history was provided, so purchase history is being "
            "used as a proxy for demand. Ordering is lumpier than consumption, "
            "which inflates forecast error and therefore safety stock. Treat "
            "these recommendations as directional until usage data is available."
        )

    if items_df is not None:
        items_df = items_df.drop_duplicates(subset="item", keep="first")
    else:
        warnings.append(
            "No item catalog was provided. Unit costs, lead times, shelf lives "
            "and case packs fall back to business-profile defaults, so dollar "
            "figures are indicative only."
        )

    dataset = Dataset(
        profile=profile,
        sales=sales_df,
        purchases=purchases_df,
        items=items_df,
        demand_source=demand_source,
        warnings=warnings,
    )
    warnings.extend(_coverage_warnings(dataset))
    return dataset


def _derive_unit_cost(purchases: pd.DataFrame) -> pd.DataFrame:
    """Fill ``unit_cost`` from ``extended_cost / quantity`` where possible."""
    frame = purchases.copy()
    if "unit_cost" not in frame.columns:
        frame["unit_cost"] = pd.NA
    if "extended_cost" in frame.columns:
        derived = frame["extended_cost"] / frame["quantity"].replace(0, pd.NA)
        frame["unit_cost"] = pd.to_numeric(frame["unit_cost"], errors="coerce").fillna(derived)
    frame["unit_cost"] = pd.to_numeric(frame["unit_cost"], errors="coerce")
    return frame


def _coverage_warnings(dataset: Dataset) -> list[str]:
    """Flag data-quality problems that change how results should be read."""
    warnings: list[str] = []
    span = dataset.date_range()
    if span is not None:
        days = (span[1] - span[0]).days + 1
        if days < 28:
            warnings.append(
                f"Only {days} days of history are available. Day-of-week "
                "seasonality needs at least four weeks to estimate reliably; "
                "the forecast will fall back to simpler models."
            )
        elif days < 56:
            warnings.append(
                f"{days} days of history is workable but thin. Accuracy and "
                "safety-stock estimates improve noticeably past ~12 weeks."
            )

    if dataset.items is not None and dataset.sales is not None:
        catalogued = set(dataset.items["item"])
        seen = set(dataset.sales["item"])
        uncatalogued = sorted(seen - catalogued)
        if uncatalogued:
            shown = ", ".join(uncatalogued[:5])
            more = "" if len(uncatalogued) <= 5 else f" (+{len(uncatalogued) - 5} more)"
            warnings.append(
                f"{len(uncatalogued)} item(s) with usage history are not in the "
                f"catalog and will use default costs: {shown}{more}."
            )
    return warnings


def load_sample_dataset(profile: BusinessProfile | None = None) -> Dataset:
    """Load the bundled synthetic coffee-shop dataset.

    Regenerate it with ``python data/generate_sample_data.py``.
    """
    missing = [
        name
        for name in (
            "coffee_shop_sales.csv",
            "coffee_shop_purchases.csv",
            "coffee_shop_items.csv",
        )
        if not (SAMPLE_DIR / name).exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"Sample data missing: {', '.join(missing)}. "
            f"Run `python data/generate_sample_data.py` to regenerate it."
        )

    return build_dataset(
        profile=profile,
        sales=SAMPLE_DIR / "coffee_shop_sales.csv",
        purchases=SAMPLE_DIR / "coffee_shop_purchases.csv",
        items=SAMPLE_DIR / "coffee_shop_items.csv",
    )
