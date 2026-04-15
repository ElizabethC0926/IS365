"""
tests/test_tax_engine.py
------------------------
Unit tests for tax_engine.py.

Each test targets a specific rule boundary:
  - Einkommensteuer: Grundfreibetrag, zone transitions, Splitting, Soli
  - AfA: rates by building type and year
  - 15% rule: window/amount boundaries
  - Rent rule: 66%/50% thresholds
  - §23: 10-year boundary
  - Grunderwerbsteuer: state rates
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from tax_engine import TaxEngine

PARAMS = "tax_params.json"


@pytest.fixture(scope="module")
def tax():
    return TaxEngine(PARAMS)


# ─────────────────────────────────────────────────────────────────────────────
# Einkommensteuer §32a
# ─────────────────────────────────────────────────────────────────────────────

class TestEinkommensteuer:

    def test_below_grundfreibetrag_is_zero(self, tax):
        """Income at or below Grundfreibetrag → zero tax."""
        r = tax.calc_income_tax(12096, 2025, "single")
        assert r["einkommensteuer"] == 0.0
        assert r["solidaritaetszuschlag"] == 0.0

    def test_just_above_grundfreibetrag(self, tax):
        """One euro above Grundfreibetrag → small positive tax."""
        r = tax.calc_income_tax(12097, 2025, "single")
        assert r["einkommensteuer"] > 0.0

    def test_zone2_rate_is_42_pct(self, tax):
        """Income well inside zone 2 → marginal rate 42%."""
        r = tax.calc_income_tax(150_000, 2025, "single")
        assert r["marginal_rate"] == pytest.approx(0.42, abs=0.001)

    def test_zone3_rate_is_45_pct(self, tax):
        """Income above 277,826 → marginal rate 45%."""
        r = tax.calc_income_tax(300_000, 2025, "single")
        assert r["marginal_rate"] == pytest.approx(0.45, abs=0.001)

    def test_splitting_lower_than_single(self, tax):
        """Married filing always produces lower or equal tax than single."""
        single  = tax.calc_income_tax(100_000, 2025, "single")
        married = tax.calc_income_tax(100_000, 2025, "married")
        assert married["total_tax"] < single["total_tax"]

    def test_splitting_symmetry(self, tax):
        """Splitting: tax(200k married) == 2 × tax(100k single)."""
        married = tax.calc_income_tax(200_000, 2025, "married")
        single  = tax.calc_income_tax(100_000, 2025, "single")
        assert married["einkommensteuer"] == pytest.approx(
            2 * single["einkommensteuer"], rel=0.001
        )

    def test_effective_rate_increases_with_income(self, tax):
        """Progressive: effective rate at 100k > effective rate at 50k."""
        r50  = tax.calc_income_tax(50_000,  2025, "single")
        r100 = tax.calc_income_tax(100_000, 2025, "single")
        assert r100["effective_rate"] > r50["effective_rate"]

    def test_2026_grundfreibetrag_higher(self, tax):
        """2026 Grundfreibetrag (12,348) is higher than 2025 (12,096)."""
        r25 = tax.calc_income_tax(12_200, 2025, "single")
        r26 = tax.calc_income_tax(12_200, 2026, "single")
        # 12,200 is above 2025 threshold but below 2026 threshold
        assert r25["einkommensteuer"] > 0
        assert r26["einkommensteuer"] == 0.0

    def test_future_year_fallback_to_2026(self, tax):
        """Year 2030 (not in params) falls back to 2026 — no KeyError."""
        r = tax.calc_income_tax(80_000, 2030, "single")
        r26 = tax.calc_income_tax(80_000, 2026, "single")
        assert r["einkommensteuer"] == r26["einkommensteuer"]

    def test_zero_income_returns_zero(self, tax):
        """Zero income → zero tax, no division errors."""
        r = tax.calc_income_tax(0, 2025, "single")
        assert r["einkommensteuer"] == 0.0
        assert r["effective_rate"] == 0.0

    def test_soli_not_triggered_below_freigrenze(self, tax):
        """Soli Freigrenze 2025 single = 18,130 ESt. Below → 0 Soli."""
        # Income that produces ESt just below 18,130
        r = tax.calc_income_tax(50_000, 2025, "single")
        # At 50k the ESt is around 12,000 — below Freigrenze
        assert r["solidaritaetszuschlag"] == 0.0

    def test_soli_triggered_above_freigrenze(self, tax):
        """High income → Soli > 0."""
        r = tax.calc_income_tax(200_000, 2025, "single")
        assert r["solidaritaetszuschlag"] > 0.0

    def test_returns_law_ref(self, tax):
        r = tax.calc_income_tax(80_000, 2025, "single")
        assert "§32a" in r["law_ref"]


# ─────────────────────────────────────────────────────────────────────────────
# AfA §7
# ─────────────────────────────────────────────────────────────────────────────

class TestAfA:

    def test_standard_building_rate_2pct(self, tax):
        """Standard building (pre-2023) AfA rate = 2%."""
        r = tax.calc_afa(300_000, 0, 2020, 2025, "standard")
        assert r["building_afa"] == pytest.approx(6_000, abs=1)
        assert r["building_rate_used"] == 0.02

    def test_neubau_post_2023_rate_3pct(self, tax):
        """Neubau post-2023 → 3% rate."""
        r = tax.calc_afa(300_000, 0, 2024, 2025, "neubau_post_2023")
        assert r["building_afa"] == pytest.approx(9_000, abs=1)
        assert r["building_rate_used"] == 0.03

    def test_movable_deductible_in_first_5_years(self, tax):
        """Movables (kitchen) deductible at 20%/yr in years 1–5."""
        r = tax.calc_afa(200_000, 15_000, 2020, 2023, "standard")
        assert r["movable_afa"] == pytest.approx(3_000, abs=1)   # 15k × 20%

    def test_movable_not_deductible_after_5_years(self, tax):
        """Movables AfA stops after 5 years (fully depreciated)."""
        r = tax.calc_afa(200_000, 15_000, 2018, 2025, "standard")
        assert r["movable_afa"] == 0.0

    def test_denkmal_phase_1_rate(self, tax):
        """Denkmal years 1–8: 9% rate."""
        r = tax.calc_afa(200_000, 0, 2020, 2023, "denkmal")
        assert r["building_rate_used"] == pytest.approx(0.09, abs=0.001)

    def test_denkmal_phase_2_rate(self, tax):
        """Denkmal years 9–12: 7% rate."""
        r = tax.calc_afa(200_000, 0, 2014, 2025, "denkmal")  # 11 years held
        assert r["building_rate_used"] == pytest.approx(0.07, abs=0.001)

    def test_zero_building_value_gives_zero_afa(self, tax):
        r = tax.calc_afa(0, 0, 2020, 2025, "standard")
        assert r["building_afa"] == 0.0
        assert r["total_afa"] == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 15% Rule §6 Abs.1 Nr.1a
# ─────────────────────────────────────────────────────────────────────────────

class TestRule15Pct:

    def test_triggered_in_window_above_limit(self, tax):
        """Year 2, spend > 15% of price → triggered."""
        r = tax.check_15pct_rule(65_000, 400_000, 2, 2025)
        assert r["triggered"] is True
        assert r["limit_amount"] == pytest.approx(60_000, abs=1)

    def test_not_triggered_outside_window(self, tax):
        """Year 4 (outside 3-year window) → not triggered even if > 15%."""
        r = tax.check_15pct_rule(65_000, 400_000, 4, 2025)
        assert r["triggered"] is False

    def test_not_triggered_below_limit(self, tax):
        """Year 2, spend < 15% → not triggered."""
        r = tax.check_15pct_rule(50_000, 400_000, 2, 2025)
        assert r["triggered"] is False
        assert r["limit_amount"] == pytest.approx(60_000, abs=1)

    def test_boundary_exactly_at_limit(self, tax):
        """Spend exactly equal to 15% limit → not triggered (must EXCEED)."""
        r = tax.check_15pct_rule(60_000, 400_000, 2, 2025)
        assert r["triggered"] is False

    def test_triggered_in_year_3(self, tax):
        """Year 3 is still within the 3-year window."""
        r = tax.check_15pct_rule(65_000, 400_000, 3, 2025)
        assert r["triggered"] is True

    def test_law_ref_present(self, tax):
        r = tax.check_15pct_rule(10_000, 400_000, 1, 2025)
        assert "§6" in r["law_ref"]


# ─────────────────────────────────────────────────────────────────────────────
# Rent Rule §21 Abs.2
# ─────────────────────────────────────────────────────────────────────────────

class TestRentRule:

    def test_full_deduction_at_market(self, tax):
        """Rent at 100% of market → full deduction ratio 1.0."""
        r = tax.check_rent_rule(18_000, 18_000, 2025)
        assert r["deduction_ratio"] == 1.0
        assert r["zone"] == "full"
        assert r["flag_rent_too_low"] is False

    def test_full_deduction_at_66pct(self, tax):
        """Rent exactly at 66% threshold → still full deduction."""
        r = tax.check_rent_rule(11_880, 18_000, 2025)  # 66% of 18k
        assert r["deduction_ratio"] == 1.0
        assert r["zone"] == "full"

    def test_half_deduction_below_66pct(self, tax):
        """Rent just below 66% → half deduction."""
        r = tax.check_rent_rule(11_000, 18_000, 2025)  # ~61%
        assert r["deduction_ratio"] == 0.5
        assert r["zone"] == "half"
        assert r["flag_rent_too_low"] is True

    def test_no_deduction_below_50pct(self, tax):
        """Rent below 50% → no deduction."""
        r = tax.check_rent_rule(8_000, 18_000, 2025)   # ~44%
        assert r["deduction_ratio"] == 0.0
        assert r["zone"] == "none"
        assert r["flag_rent_too_low"] is True

    def test_zero_market_rent_defaults_to_full(self, tax):
        """market_rent=0 → no comparison possible → full ratio."""
        r = tax.check_rent_rule(5_000, 0, 2025)
        assert r["deduction_ratio"] == 1.0


# ─────────────────────────────────────────────────────────────────────────────
# §23 Speculation Tax
# ─────────────────────────────────────────────────────────────────────────────

class TestSpeculationTax:

    def test_tax_free_at_10_years(self, tax):
        """Exactly 10 years held → tax-free."""
        r = tax.calc_speculation_tax(500_000, 400_000, 40_000, 10,
                                     80_000, "single", 2025)
        assert r["tax_free"] is True
        assert r["speculation_tax"] == 0.0

    def test_tax_free_above_10_years(self, tax):
        """11 years → also tax-free."""
        r = tax.calc_speculation_tax(500_000, 400_000, 40_000, 11,
                                     80_000, "single", 2025)
        assert r["tax_free"] is True

    def test_taxable_below_10_years(self, tax):
        """9 years → taxable."""
        r = tax.calc_speculation_tax(500_000, 400_000, 40_000, 9,
                                     80_000, "single", 2025)
        assert r["tax_free"] is False
        assert r["speculation_tax"] > 0.0

    def test_afa_recapture_increases_gain(self, tax):
        """More cumulative AfA → higher taxable gain (AfA recapture)."""
        r_low  = tax.calc_speculation_tax(500_000, 400_000, 10_000, 5,
                                          80_000, "single", 2025)
        r_high = tax.calc_speculation_tax(500_000, 400_000, 50_000, 5,
                                          80_000, "single", 2025)
        assert r_high["taxable_gain"] > r_low["taxable_gain"]

    def test_no_gain_no_tax(self, tax):
        """Sale price = purchase price - AfA → zero or minimal gain."""
        r = tax.calc_speculation_tax(360_000, 400_000, 40_000, 5,
                                     80_000, "single", 2025)
        assert r["taxable_gain"] == 0.0
        assert r["speculation_tax"] == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Grunderwerbsteuer
# ─────────────────────────────────────────────────────────────────────────────

class TestGrundeErwerbsteuer:

    def test_bayern_rate(self, tax):
        r = tax.calc_grunderwerbsteuer(400_000, "Bayern", 2025)
        assert r["rate"] == pytest.approx(0.035, abs=0.001)
        assert r["grunderwerbsteuer"] == pytest.approx(14_000, abs=1)

    def test_berlin_rate(self, tax):
        r = tax.calc_grunderwerbsteuer(400_000, "Berlin", 2025)
        assert r["rate"] == pytest.approx(0.06, abs=0.001)
        assert r["grunderwerbsteuer"] == pytest.approx(24_000, abs=1)

    def test_unknown_state_uses_default(self, tax):
        """Unknown state falls back to default rate (5%)."""
        r = tax.calc_grunderwerbsteuer(400_000, "Unbekannt", 2025)
        assert r["rate"] == pytest.approx(0.05, abs=0.001)
