"""Ingestion: column mapping, cleaning, validation and dataset assembly."""

from __future__ import annotations

import io

import pandas as pd
import pytest

from core.business_profile import BusinessProfile
from core.data_loader import build_dataset, read_table
from core.schemas import ITEMS_SCHEMA, SALES_SCHEMA, SchemaError, normalize_header


class TestHeaderNormalization:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("Item Name ", "item_name"),
            ("QTY.", "qty"),
            ("Unit-Cost", "unit_cost"),
            ("  date  ", "date"),
            ("Order   Date", "order_date"),
        ],
    )
    def test_normalizes_messy_headers(self, raw, expected):
        assert normalize_header(raw) == expected


class TestColumnMapping:
    def test_accepts_real_world_aliases(self):
        csv = io.StringIO("Order Date,Product,Qty\n2026-01-01,Milk,3\n2026-01-02,Milk,4\n")
        frame = read_table(csv, SALES_SCHEMA)
        assert list(frame.columns) == ["date", "item", "quantity"]
        assert frame["quantity"].tolist() == [3, 4]

    def test_ignores_columns_it_does_not_recognise(self):
        csv = io.StringIO("date,item,quantity,register_id\n2026-01-01,Milk,3,R2\n")
        assert "register_id" not in read_table(csv, SALES_SCHEMA).columns

    def test_missing_required_column_names_the_problem(self):
        csv = io.StringIO("date,quantity\n2026-01-01,3\n")
        with pytest.raises(SchemaError, match="item"):
            read_table(csv, SALES_SCHEMA)

    def test_first_matching_header_wins(self):
        csv = io.StringIO("date,item,quantity,qty\n2026-01-01,Milk,3,99\n")
        assert read_table(csv, SALES_SCHEMA)["quantity"].iloc[0] == 3


class TestCleaning:
    def test_drops_rows_with_unparseable_required_values(self):
        csv = io.StringIO(
            "date,item,quantity\n2026-01-01,Milk,3\nnot-a-date,Milk,4\n2026-01-03,Milk,oops\n"
        )
        assert len(read_table(csv, SALES_SCHEMA)) == 1

    def test_trims_and_collapses_whitespace_in_item_names(self):
        csv = io.StringIO("date,item,quantity\n2026-01-01,  Whole   Milk ,3\n")
        assert read_table(csv, SALES_SCHEMA)["item"].iloc[0] == "Whole Milk"

    def test_all_rows_unusable_is_an_error(self):
        csv = io.StringIO("date,item,quantity\nbad,,\n")
        with pytest.raises(SchemaError):
            read_table(csv, SALES_SCHEMA)

    def test_empty_file_is_an_error(self):
        with pytest.raises(SchemaError):
            read_table(pd.DataFrame(columns=["date", "item", "quantity"]), SALES_SCHEMA)


class TestDatasetAssembly:
    def test_requires_a_demand_source(self):
        with pytest.raises(SchemaError):
            build_dataset(profile=BusinessProfile())

    def test_purchases_alone_become_a_proxy_with_a_warning(self):
        purchases = pd.DataFrame(
            {
                "date": pd.date_range("2026-01-01", periods=10, freq="D"),
                "item": ["Milk"] * 10,
                "quantity": [4.0] * 10,
                "unit_cost": [4.25] * 10,
            }
        )
        dataset = build_dataset(profile=BusinessProfile(), purchases=purchases)
        assert dataset.demand_source == "purchases_proxy"
        assert any("proxy" in w for w in dataset.warnings)

    def test_derives_unit_cost_from_an_extended_cost_column(self):
        purchases = pd.DataFrame(
            {
                "date": ["2026-01-01"],
                "item": ["Milk"],
                "quantity": [4.0],
                "line_total": [17.0],
            }
        )
        dataset = build_dataset(profile=BusinessProfile(), purchases=purchases)
        assert dataset.purchases["unit_cost"].iloc[0] == pytest.approx(4.25)

    def test_warns_about_items_missing_from_the_catalog(self, tiny_dataset):
        extra = pd.concat(
            [
                tiny_dataset.sales,
                pd.DataFrame([{"date": pd.Timestamp("2026-01-01"), "item": "Sprocket", "quantity": 1.0}]),
            ]
        )
        dataset = build_dataset(
            profile=BusinessProfile(), sales=extra, items=tiny_dataset.items
        )
        assert any("Sprocket" in w for w in dataset.warnings)

    def test_warns_when_history_is_too_short(self):
        sales = pd.DataFrame(
            {
                "date": pd.date_range("2026-01-01", periods=10, freq="D"),
                "item": ["Milk"] * 10,
                "quantity": [4.0] * 10,
            }
        )
        dataset = build_dataset(profile=BusinessProfile(), sales=sales)
        assert any("days of history" in w for w in dataset.warnings)


class TestDemandFrame:
    def test_fills_missing_days_with_zero_demand(self):
        sales = pd.DataFrame(
            {
                "date": ["2026-01-01", "2026-01-05"],
                "item": ["Milk", "Milk"],
                "quantity": [3.0, 5.0],
            }
        )
        dataset = build_dataset(profile=BusinessProfile(), sales=sales)
        series = dataset.demand_series("Milk")
        assert len(series) == 5
        assert series.iloc[1:4].sum() == 0.0

    def test_does_not_backfill_before_an_item_first_appears(self):
        sales = pd.DataFrame(
            {
                "date": ["2026-01-01", "2026-01-10", "2026-01-10"],
                "item": ["Milk", "Milk", "Matcha"],
                "quantity": [3.0, 5.0, 1.0],
            }
        )
        dataset = build_dataset(profile=BusinessProfile(), sales=sales)
        assert len(dataset.demand_series("Milk")) == 10
        assert len(dataset.demand_series("Matcha")) == 1

    def test_sums_multiple_rows_for_the_same_day(self):
        sales = pd.DataFrame(
            {
                "date": ["2026-01-01", "2026-01-01"],
                "item": ["Milk", "Milk"],
                "quantity": [3.0, 4.0],
            }
        )
        dataset = build_dataset(profile=BusinessProfile(), sales=sales)
        assert dataset.demand_series("Milk").iloc[0] == 7.0

    def test_returns_and_waste_cannot_push_demand_negative(self):
        sales = pd.DataFrame(
            {
                "date": ["2026-01-01", "2026-01-01"],
                "item": ["Milk", "Milk"],
                "quantity": [3.0, -8.0],
            }
        )
        dataset = build_dataset(profile=BusinessProfile(), sales=sales)
        assert dataset.demand_series("Milk").iloc[0] == 0.0


class TestItemAttributes:
    def test_reads_the_catalog(self, tiny_dataset):
        attributes = tiny_dataset.attributes("Gadget")
        assert attributes.unit_cost == 5.0
        assert attributes.lead_time_days == 2
        assert attributes.case_pack == 6
        assert attributes.is_perishable

    def test_blank_shelf_life_means_not_perishable(self, tiny_dataset):
        assert not tiny_dataset.attributes("Widget").is_perishable

    def test_falls_back_to_profile_defaults_and_records_what_was_inferred(self):
        sales = pd.DataFrame(
            {
                "date": pd.date_range("2026-01-01", periods=30, freq="D"),
                "item": ["Mystery"] * 30,
                "quantity": [2.0] * 30,
            }
        )
        profile = BusinessProfile(default_lead_time_days=9.0)
        dataset = build_dataset(profile=profile, sales=sales)
        attributes = dataset.attributes("Mystery")
        assert attributes.lead_time_days == 9.0
        assert "lead_time_days" in attributes.inferred_fields

    def test_falls_back_to_the_average_cost_actually_paid(self):
        sales = pd.DataFrame(
            {
                "date": pd.date_range("2026-01-01", periods=30, freq="D"),
                "item": ["Milk"] * 30,
                "quantity": [4.0] * 30,
            }
        )
        purchases = pd.DataFrame(
            {"date": ["2026-01-01"], "item": ["Milk"], "quantity": [10.0], "unit_cost": [3.5]}
        )
        dataset = build_dataset(profile=BusinessProfile(), sales=sales, purchases=purchases)
        assert dataset.attributes("Milk").unit_cost == pytest.approx(3.5)

    def test_margin_derives_from_sell_price_when_given(self):
        sales = pd.DataFrame(
            {
                "date": pd.date_range("2026-01-01", periods=30, freq="D"),
                "item": ["Scone"] * 30,
                "quantity": [6.0] * 30,
            }
        )
        items = pd.DataFrame([{"item": "Scone", "unit_cost": 1.0, "sell_price": 4.0}])
        dataset = build_dataset(profile=BusinessProfile(), sales=sales, items=items)
        assert dataset.attributes("Scone").margin_per_unit == pytest.approx(3.0)


class TestSchemaIntrospection:
    def test_items_catalog_only_requires_the_item_column(self):
        assert ITEMS_SCHEMA.required_columns == ("item",)

    def test_a_bare_catalog_loads(self):
        csv = io.StringIO("item\nMilk\nBeans\n")
        assert len(read_table(csv, ITEMS_SCHEMA)) == 2
