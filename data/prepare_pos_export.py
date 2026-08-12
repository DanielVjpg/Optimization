"""Convert a raw POS transaction export into the `sales` table the app expects.

Real POS exports are transaction-level: one row per line item per sale, often
across several locations, with whatever delimiter the vendor felt like using.
The inventory module wants one row per item per day for a single site. This
script does that reduction.

Usage::

    python data/prepare_pos_export.py raw_export.csv
    python data/prepare_pos_export.py raw_export.csv --store "Astoria"
    python data/prepare_pos_export.py raw_export.csv --group-by product_type
    python data/prepare_pos_export.py raw_export.csv --category Bakery "Coffee beans"

By default it keeps the location with the most transaction rows, groups by
``product_detail``, and writes ``data/prepared/<store>_sales.csv``.

The delimiter is sniffed rather than assumed — the export this was written
against is pipe-delimited, and a comma-assuming reader silently parses the
whole header as a single column, which surfaces later as a confusing
"missing required columns" error.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "prepared"

# Column names in the source export. Override from the command line when a
# different POS uses different names.
DATE_COLUMN = "transaction_date"
QUANTITY_COLUMN = "transaction_qty"
STORE_COLUMN = "store_location"
ITEM_COLUMN = "product_detail"
CATEGORY_COLUMN = "product_category"
PRICE_COLUMN = "unit_price"


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_") or "store"


def normalize_items(values: pd.Series) -> pd.Series:
    """Trim and collapse whitespace in item names.

    Not cosmetic. Real exports carry names like ``"Scottish Cream Scone "``
    alongside the untrimmed version, and treating those as two products splits
    one item's demand history in half — which quietly halves its forecast and
    its reorder point.
    """
    return values.astype(str).str.strip().str.replace(r"\s+", " ", regex=True)


CANDIDATE_DELIMITERS = (",", "|", ";", "\t")


def detect_delimiter(path: Path, sample_lines: int = 20) -> str:
    """Pick the delimiter that splits the header and body consistently.

    Deliberately not ``csv.Sniffer`` (or pandas' ``sep=None``, which wraps it):
    on this project's first real export it chose ``t`` — the letter — because
    it appears regularly inside words like "transaction". Counting explicit
    candidates and requiring the split to be stable across rows is duller and
    does not fail that way.
    """
    with open(path, "r", encoding="utf-8-sig", errors="replace") as handle:
        lines = [line.rstrip("\n") for _, line in zip(range(sample_lines), handle)]
    lines = [line for line in lines if line.strip()]
    if not lines:
        raise SystemExit(f"{path} is empty.")

    best, best_columns = None, 1
    for delimiter in CANDIDATE_DELIMITERS:
        counts = [line.count(delimiter) for line in lines]
        columns = counts[0] + 1
        # The header must split, and every sampled row must split the same way.
        if columns > 1 and len(set(counts)) == 1 and columns > best_columns:
            best, best_columns = delimiter, columns

    if best is None:
        raise SystemExit(
            f"Could not detect a delimiter for {path}. Tried "
            f"{', '.join(repr(d) for d in CANDIDATE_DELIMITERS)}. "
            f"Header reads: {lines[0][:200]!r}\nPass one with --delimiter."
        )
    return best


def read_export(path: Path, delimiter: str | None = None) -> pd.DataFrame:
    """Read a delimited export, detecting the separator when not given."""
    separator = delimiter or detect_delimiter(path)
    frame = pd.read_csv(path, sep=separator, engine="python")
    if frame.shape[1] == 1:
        raise SystemExit(
            f"{path} parsed as a single column using {separator!r}. "
            f"The header read as: {frame.columns[0]!r}\nPass --delimiter."
        )
    return frame


def require_columns(frame: pd.DataFrame, columns: dict[str, str]) -> None:
    missing = {label: name for label, name in columns.items() if name not in frame.columns}
    if missing:
        raise SystemExit(
            "The export is missing expected column(s): "
            + ", ".join(f"{name} (for {label})" for label, name in missing.items())
            + f".\nFound: {', '.join(map(str, frame.columns))}"
            + "\nPass the right names with --date-column / --qty-column / "
            "--store-column / --group-by."
        )


def prepare(
    frame: pd.DataFrame,
    *,
    store: str | None,
    item_column: str,
    date_column: str,
    quantity_column: str,
    store_column: str | None,
    categories: list[str] | None,
    category_column: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Filter to one store and aggregate to one row per item per day."""
    report: dict[str, object] = {"rows_in": len(frame)}

    # --- pick the store ------------------------------------------------
    if store_column and store_column in frame.columns:
        counts = frame[store_column].value_counts()
        report["stores_available"] = counts.to_dict()
        if store is None:
            store = str(counts.index[0])
        elif store not in set(counts.index):
            raise SystemExit(
                f"No location named {store!r}. Available: "
                + ", ".join(map(str, counts.index))
            )
        frame = frame[frame[store_column] == store]
        report["store"] = store
    else:
        report["store"] = None

    # --- optional category filter --------------------------------------
    if categories:
        if category_column not in frame.columns:
            raise SystemExit(f"--category needs a {category_column!r} column.")
        known = set(frame[category_column].unique())
        unknown = [c for c in categories if c not in known]
        if unknown:
            raise SystemExit(
                f"Unknown categor(ies): {', '.join(unknown)}. "
                f"Available: {', '.join(sorted(known))}"
            )
        frame = frame[frame[category_column].isin(categories)]
        report["categories"] = categories

    if frame.empty:
        raise SystemExit("No rows left after filtering.")

    # --- clean ----------------------------------------------------------
    working = frame.loc[:, [date_column, item_column, quantity_column]].copy()
    working[date_column] = pd.to_datetime(working[date_column], errors="coerce").dt.normalize()
    working[quantity_column] = pd.to_numeric(working[quantity_column], errors="coerce")
    raw_names = working[item_column].nunique()
    working[item_column] = normalize_items(working[item_column])
    report["names_merged_by_cleaning"] = raw_names - working[item_column].nunique()

    before = len(working)
    working = working.dropna(subset=[date_column, item_column, quantity_column])
    report["rows_dropped_unparseable"] = before - len(working)

    # --- aggregate ------------------------------------------------------
    daily = (
        working.groupby([date_column, item_column], as_index=False)[quantity_column]
        .sum()
        .rename(
            columns={date_column: "date", item_column: "item", quantity_column: "quantity"}
        )
        .sort_values(["date", "item"])
        .reset_index(drop=True)
    )
    daily["date"] = daily["date"].dt.strftime("%Y-%m-%d")

    report.update(
        {
            "rows_out": len(daily),
            "items": daily["item"].nunique(),
            "date_start": daily["date"].min(),
            "date_end": daily["date"].max(),
            "days_covered": daily["date"].nunique(),
            "total_units": float(daily["quantity"].sum()),
        }
    )
    return daily, report


def build_catalog(frame: pd.DataFrame, daily: pd.DataFrame, item_column: str) -> pd.DataFrame | None:
    """Optional item catalog seeded with the average selling price.

    Cost, lead time, shelf life and case pack cannot be derived from sales
    data — those come from invoices and the supplier. This gets the item list
    and menu price in place so the rest can be filled in by hand.
    """
    if PRICE_COLUMN not in frame.columns:
        return None
    priced = frame.loc[:, [item_column, PRICE_COLUMN]].copy()
    # Same normalization as the sales table, or the two files disagree on
    # which items exist and the catalog silently misses one.
    priced[item_column] = normalize_items(priced[item_column])
    prices = priced.groupby(item_column)[PRICE_COLUMN].mean().round(2)
    catalog = pd.DataFrame({"item": prices.index, "sell_price": prices.to_numpy()})
    catalog = catalog[catalog["item"].isin(set(daily["item"]))].sort_values("item")
    for column in ("unit_cost", "lead_time_days", "shelf_life_days", "case_pack", "min_order_qty"):
        catalog[column] = ""
    return catalog.reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path, help="Raw POS export (any common delimiter)")
    parser.add_argument("--store", default=None, help="Location to keep (default: the busiest)")
    parser.add_argument("--group-by", default=ITEM_COLUMN, help=f"Item column (default: {ITEM_COLUMN})")
    parser.add_argument("--date-column", default=DATE_COLUMN)
    parser.add_argument("--qty-column", default=QUANTITY_COLUMN)
    parser.add_argument("--store-column", default=STORE_COLUMN)
    parser.add_argument("--category-column", default=CATEGORY_COLUMN)
    parser.add_argument("--category", nargs="+", default=None, help="Keep only these categories")
    parser.add_argument("--delimiter", default=None, help="Force a delimiter instead of detecting it")
    parser.add_argument("--out", type=Path, default=None, help="Output CSV path")
    parser.add_argument("--catalog", action="store_true", help="Also write a starter item catalog")
    args = parser.parse_args(argv)

    if not args.source.exists():
        raise SystemExit(f"No such file: {args.source}")

    frame = read_export(args.source, args.delimiter)
    require_columns(
        frame,
        {"date": args.date_column, "quantity": args.qty_column, "item": args.group_by},
    )

    daily, report = prepare(
        frame,
        store=args.store,
        item_column=args.group_by,
        date_column=args.date_column,
        quantity_column=args.qty_column,
        store_column=args.store_column,
        categories=args.category,
        category_column=args.category_column,
    )

    output = args.out
    if output is None:
        DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output = DEFAULT_OUTPUT_DIR / f"{slugify(report.get('store') or 'all')}_sales.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(output, index=False)

    # --- report ---------------------------------------------------------
    print(f"Read     {report['rows_in']:,} transaction rows from {args.source.name}")
    if report.get("stores_available"):
        for name, count in report["stores_available"].items():
            marker = "  <- kept" if name == report.get("store") else ""
            print(f"           {name:<20} {count:>8,}{marker}")
    if report.get("categories"):
        print(f"Filtered to categories: {', '.join(report['categories'])}")
    if report["rows_dropped_unparseable"]:
        print(f"Dropped  {report['rows_dropped_unparseable']:,} rows with unparseable date/item/quantity")
    if report.get("names_merged_by_cleaning"):
        print(
            f"Merged   {report['names_merged_by_cleaning']} item name(s) that differed "
            "only by whitespace"
        )
    print(f"Grouped  by {args.group_by}")
    print(
        f"Wrote    {report['rows_out']:,} rows · {report['items']} items · "
        f"{report['date_start']} → {report['date_end']} "
        f"({report['days_covered']} days) · {report['total_units']:,.0f} units"
    )
    print(f"Output   {output}  ({output.stat().st_size / 1024:,.0f} KB)")

    if args.catalog:
        catalog = build_catalog(frame, daily, args.group_by)
        if catalog is None:
            print("Skipped  catalog: no unit_price column in the export")
        else:
            catalog_path = output.with_name(output.stem.replace("_sales", "") + "_items.csv")
            catalog.to_csv(catalog_path, index=False)
            print(f"Output   {catalog_path}  ({len(catalog)} items, costs left blank to fill in)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
