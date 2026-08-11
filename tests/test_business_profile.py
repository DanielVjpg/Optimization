"""Business profile: validation, the safety factor, and YAML loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.business_profile import BusinessProfile

SAMPLE_PROFILE = Path(__file__).resolve().parent.parent / "data" / "sample_business_profile.yaml"


class TestSafetyFactor:
    @pytest.mark.parametrize(
        "service_level,expected",
        [(0.90, 1.282), (0.95, 1.645), (0.975, 1.960), (0.99, 2.326)],
    )
    def test_z_matches_the_standard_normal_table(self, service_level, expected):
        assert BusinessProfile(service_level=service_level).z == pytest.approx(expected, abs=1e-3)

    def test_z_increases_with_the_service_level(self):
        assert BusinessProfile(service_level=0.99).z > BusinessProfile(service_level=0.90).z


class TestValidation:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"service_level": 1.0},
            {"service_level": 0.1},
            {"holding_cost_rate": 0.0},
            {"order_fixed_cost": -1.0},
            {"days_per_year": 0},
            {"shelf_life_utilization": 0.0},
            {"shelf_life_utilization": 1.5},
            {"stockout_realization_rate": 1.5},
            {"max_days_of_cover": 0},
        ],
    )
    def test_rejects_impossible_assumptions(self, kwargs):
        with pytest.raises(ValueError):
            BusinessProfile(**kwargs)


class TestSerialization:
    def test_round_trips_through_a_dict(self):
        original = BusinessProfile(name="Round Trip", service_level=0.97)
        restored = BusinessProfile.from_dict(original.to_dict())
        assert restored.name == "Round Trip"
        assert restored.service_level == 0.97

    def test_unknown_keys_are_preserved_not_rejected(self):
        """A client profile may carry parameters for modules that don't exist
        yet; dropping them silently would lose real configuration."""
        profile = BusinessProfile.from_dict({"name": "Future", "seating_capacity": 42})
        assert profile.extra["seating_capacity"] == 42

    def test_loads_the_sample_yaml_profile(self):
        profile = BusinessProfile.from_yaml(SAMPLE_PROFILE)
        assert profile.name == "Sample Coffee Shop"
        assert profile.order_fixed_cost == pytest.approx(4.0)
        assert profile.shelf_life_utilization == pytest.approx(0.5)
        assert profile.labor_rate_per_hour == pytest.approx(18.0)

    def test_money_formatting(self):
        assert BusinessProfile().money(1234.5) == "$1,234.50"
