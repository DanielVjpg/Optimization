"""The POS export converter.

Covers the two failures that a real export actually hit: a non-comma delimiter
read as a single column, and item names differing only by whitespace.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Loaded by path: data/ is a data directory, not an importable package.
_spec = importlib.util.spec_from_file_location(
    "prepare_pos_export", PROJECT_ROOT / "data" / "prepare_pos_export.py"
)
prepare_pos_export = importlib.util.module_from_spec(_spec)
sys.modules["prepare_pos_export"] = prepare_pos_export
_spec.loader.exec_module(prepare_pos_export)


HEADER = [
    "transaction_id", "transaction_date", "transaction_qty",
    "store_location", "unit_price", "product_category", "product_detail",
]
ROWS = [
    ["1", "2023-01-01", "2", "Hell's Kitchen", "3.0", "Coffee", "Ethiopia Rg"],
    ["2", "2023-01-01", "1", "Hell's Kitchen", "3.0", "Coffee", "Ethiopia Rg"],
    ["3", "2023-01-02", "4", "Hell's Kitchen", "3.5", "Bakery", "Scottish Cream Scone "],
    ["4", "2023-01-02", "1", "Hell's Kitchen", "3.5", "Bakery", "Scottish Cream Scone"],
    ["5", "2023-01-01", "9", "Astoria", "3.0", "Coffee", "Ethiopia Rg"],
]


def write_export(path: Path, delimiter: str) -> Path:
    lines = [delimiter.join(HEADER)] + [delimiter.join(row) for row in ROWS]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


class TestDelimiterDetection:
    @pytest.mark.parametrize("delimiter", [",", "|", ";", "\t"])
    def test_detects_each_supported_delimiter(self, tmp_path, delimiter):
        source = write_export(tmp_path / "export.csv", delimiter)
        assert prepare_pos_export.detect_delimiter(source) == delimiter

    @pytest.mark.parametrize("delimiter", [",", "|", ";", "\t"])
    def test_reads_every_delimiter_into_the_same_frame(self, tmp_path, delimiter):
        source = write_export(tmp_path / "export.csv", delimiter)
        frame = prepare_pos_export.read_export(source)
        assert list(frame.columns) == HEADER
        assert len(frame) == len(ROWS)

    def test_does_not_pick_a_letter_that_appears_inside_words(self, tmp_path):
        """csv.Sniffer chose 't' on the real export, because "transaction"
        contains it at a regular cadence. Pipe must win here."""
        source = write_export(tmp_path / "export.csv", "|")
        assert prepare_pos_export.detect_delimiter(source) == "|"

    def test_rejects_a_file_with_no_usable_delimiter(self, tmp_path):
        source = tmp_path / "bad.csv"
        source.write_text("just one column\nvalue\nvalue\n", encoding="utf-8")
        with pytest.raises(SystemExit):
            prepare_pos_export.detect_delimiter(source)

    def test_an_explicit_delimiter_overrides_detection(self, tmp_path):
        source = write_export(tmp_path / "export.csv", "|")
        assert list(prepare_pos_export.read_export(source, "|").columns) == HEADER


class TestItemNameNormalization:
    def test_merges_names_differing_only_by_whitespace(self):
        values = pd.Series(["Scottish Cream Scone ", "Scottish Cream Scone", "Latte  Rg"])
        cleaned = prepare_pos_export.normalize_items(values)
        assert cleaned.iloc[0] == cleaned.iloc[1] == "Scottish Cream Scone"
        assert cleaned.iloc[2] == "Latte Rg"


class TestPrepare:
    @pytest.fixture
    def frame(self, tmp_path):
        return prepare_pos_export.read_export(write_export(tmp_path / "e.csv", "|"))

    def _prepare(self, frame, **overrides):
        kwargs = dict(
            store=None, item_column="product_detail", date_column="transaction_date",
            quantity_column="transaction_qty", store_column="store_location",
            categories=None, category_column="product_category",
        )
        kwargs.update(overrides)
        return prepare_pos_export.prepare(frame, **kwargs)

    def test_keeps_the_busiest_store_by_default(self, frame):
        daily, report = self._prepare(frame)
        assert report["store"] == "Hell's Kitchen"
        assert report["rows_in"] == 5

    def test_an_explicit_store_can_be_selected(self, frame):
        _, report = self._prepare(frame, store="Astoria")
        assert report["store"] == "Astoria"

    def test_an_unknown_store_is_rejected(self, frame):
        with pytest.raises(SystemExit):
            self._prepare(frame, store="Brooklyn")

    def test_emits_the_three_columns_the_app_expects(self, frame):
        daily, _ = self._prepare(frame)
        assert list(daily.columns) == ["date", "item", "quantity"]

    def test_sums_quantity_per_item_per_day(self, frame):
        daily, _ = self._prepare(frame)
        row = daily[(daily["date"] == "2023-01-01") & (daily["item"] == "Ethiopia Rg")]
        assert row["quantity"].iloc[0] == 3  # 2 + 1, and Astoria's 9 excluded

    def test_whitespace_variants_collapse_into_one_item(self, frame):
        daily, report = self._prepare(frame)
        scones = daily[daily["item"] == "Scottish Cream Scone"]
        assert len(scones) == 1
        assert scones["quantity"].iloc[0] == 5
        assert report["names_merged_by_cleaning"] == 1

    def test_category_filter(self, frame):
        daily, _ = self._prepare(frame, categories=["Bakery"])
        assert set(daily["item"]) == {"Scottish Cream Scone"}

    def test_an_unknown_category_is_rejected(self, frame):
        with pytest.raises(SystemExit):
            self._prepare(frame, categories=["Sandwiches"])

    def test_output_loads_through_the_apps_own_loader(self, frame, tmp_path):
        """The whole point of the script: its output must satisfy the schema
        that rejected the raw export."""
        from core.business_profile import BusinessProfile
        from core.data_loader import build_dataset

        daily, _ = self._prepare(frame)
        path = tmp_path / "sales.csv"
        daily.to_csv(path, index=False)

        dataset = build_dataset(profile=BusinessProfile(), sales=path)
        assert dataset.item_names() == ["Ethiopia Rg", "Scottish Cream Scone"]


class TestCatalog:
    def test_catalog_item_names_match_the_sales_file(self, tmp_path):
        """A catalog built without the same name cleaning silently drops the
        untrimmed item, and the app then falls back to default costs for it."""
        frame = prepare_pos_export.read_export(write_export(tmp_path / "e.csv", "|"))
        daily, _ = prepare_pos_export.prepare(
            frame, store=None, item_column="product_detail",
            date_column="transaction_date", quantity_column="transaction_qty",
            store_column="store_location", categories=None,
            category_column="product_category",
        )
        catalog = prepare_pos_export.build_catalog(frame, daily, "product_detail")
        assert set(catalog["item"]) == set(daily["item"])
        assert "sell_price" in catalog.columns

    def test_no_catalog_without_a_price_column(self, tmp_path):
        source = tmp_path / "e.csv"
        source.write_text(
            "transaction_date|transaction_qty|product_detail\n2023-01-01|2|Latte\n",
            encoding="utf-8",
        )
        frame = prepare_pos_export.read_export(source)
        assert prepare_pos_export.build_catalog(frame, pd.DataFrame(), "product_detail") is None
