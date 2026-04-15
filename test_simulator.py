"""
tests/test_simulator.py
------------------------
Integration tests for the full simulation pipeline.

Tests that the four FLAGS are correctly detected across a range of
deliberately constructed edge cases.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from simulator import run_simulation


# ─────────────────────────────────────────────────────────────────────────────
# Shared base parameters
# ─────────────────────────────────────────────────────────────────────────────

BASE = dict(
    purchase_price     = 400_000,
    state              = "Bayern",
    purchase_year      = 2024,
    personal_income    = 80_000,
    filing_status      = "single",
    market_rent_annual = 18_000,
    building_type      = "standard",
    usage              = "full_rental",
    holding_years      = 10,
    rental_ratio       = 1.0,
    renovation_year    = 5,
    renovation_amount  = 10_000,
    asset_split        = 10_000,
    land_ratio         = 0.25,
    equity_amount      = 120_000,
    annual_rate        = 0.035,
    sondertilgung_rate = 0.0,
    refi_year          = None,
    refi_rate          = None,
)


def sim(overrides: dict = {}, sale_price: float = 480_000):
    params = {**BASE, **overrides}
    return run_simulation(params, sale_price=sale_price)


# ─────────────────────────────────────────────────────────────────────────────
# FLAG_15_PERCENT_HIT
# ─────────────────────────────────────────────────────────────────────────────

class TestFlag15Pct:

    def test_triggered_when_reno_exceeds_limit_in_window(self):
        r = sim({"renovation_year": 2, "renovation_amount": 65_000})
        assert r.flags["FLAG_15_PERCENT_HIT"]["triggered"] is True

    def test_not_triggered_when_reno_outside_window(self):
        r = sim({"renovation_year": 4, "renovation_amount": 65_000})
        assert r.flags["FLAG_15_PERCENT_HIT"]["triggered"] is False

    def test_not_triggered_when_reno_below_limit(self):
        r = sim({"renovation_year": 2, "renovation_amount": 50_000})
        assert r.flags["FLAG_15_PERCENT_HIT"]["triggered"] is False

    def test_flag_records_trigger_year(self):
        r = sim({"renovation_year": 2, "renovation_amount": 65_000})
        assert r.flags["FLAG_15_PERCENT_HIT"]["year_index"] == 2

    def test_zero_renovation_never_triggers(self):
        r = sim({"renovation_year": 1, "renovation_amount": 0})
        assert r.flags["FLAG_15_PERCENT_HIT"]["triggered"] is False


# ─────────────────────────────────────────────────────────────────────────────
# FLAG_RENT_TOO_LOW
# ─────────────────────────────────────────────────────────────────────────────

class TestFlagRentTooLow:

    def test_triggered_when_rent_below_66pct(self):
        r = sim({"rental_ratio": 0.60})
        assert r.flags["FLAG_RENT_TOO_LOW"]["triggered"] is True

    def test_not_triggered_at_market_rate(self):
        r = sim({"rental_ratio": 1.0})
        assert r.flags["FLAG_RENT_TOO_LOW"]["triggered"] is False

    def test_not_triggered_exactly_at_66pct(self):
        r = sim({"rental_ratio": 0.66})
        assert r.flags["FLAG_RENT_TOO_LOW"]["triggered"] is False

    def test_triggered_below_50pct(self):
        r = sim({"rental_ratio": 0.45})
        assert r.flags["FLAG_RENT_TOO_LOW"]["triggered"] is True


# ─────────────────────────────────────────────────────────────────────────────
# FLAG_NEGATIVE_CASHFLOW
# ─────────────────────────────────────────────────────────────────────────────

class TestFlagNegativeCashflow:

    def test_triggered_when_loan_exceeds_rent(self):
        """High loan, low rent → negative cashflow in early years."""
        r = sim({
            "equity_amount": 50_000,    # small equity = big loan
            "rental_ratio":  0.70,      # below market rent
        })
        assert r.flags["FLAG_NEGATIVE_CASHFLOW"]["triggered"] is True

    def test_records_first_negative_year(self):
        r = sim({"equity_amount": 50_000})
        flag = r.flags["FLAG_NEGATIVE_CASHFLOW"]
        if flag["triggered"]:
            assert flag["year_index"] >= 1


# ─────────────────────────────────────────────────────────────────────────────
# §23 Speculation Tax via simulator
# ─────────────────────────────────────────────────────────────────────────────

class TestSpecTaxViaSimulator:

    def test_9yr_hold_has_spec_tax(self):
        r = run_simulation(
            {**BASE, "holding_years": 9},
            sale_price=500_000,
        )
        assert r.exit_result.speculation_tax_result["tax_free"] is False
        assert r.exit_result.speculation_tax_result["speculation_tax"] > 0

    def test_11yr_hold_is_tax_free(self):
        r = run_simulation(
            {**BASE, "holding_years": 11},
            sale_price=500_000,
        )
        assert r.exit_result.speculation_tax_result["tax_free"] is True
        assert r.exit_result.speculation_tax_result["speculation_tax"] == 0.0

    def test_11yr_irr_higher_than_9yr(self):
        """The §23 cliff: waiting 2 years improves IRR."""
        r9  = run_simulation({**BASE, "holding_years": 9},  sale_price=500_000)
        r11 = run_simulation({**BASE, "holding_years": 11}, sale_price=500_000)
        assert r11.irr() > r9.irr()


# ─────────────────────────────────────────────────────────────────────────────
# SimulationResult structure
# ─────────────────────────────────────────────────────────────────────────────

class TestSimulationResultStructure:

    def test_snapshot_count_matches_holding_years(self):
        r = sim({"holding_years": 10})
        assert len(r.snapshots) == 10

    def test_all_four_flags_present(self):
        r = sim()
        expected = {
            "FLAG_15_PERCENT_HIT",
            "FLAG_RENT_TOO_LOW",
            "FLAG_TAX_WASTE",
            "FLAG_NEGATIVE_CASHFLOW",
        }
        assert set(r.flags.keys()) == expected

    def test_irr_is_float_or_nan(self):
        import math
        r = sim()
        irr = r.irr()
        assert isinstance(irr, float)

    def test_cashflow_series_length(self):
        """Length = holding_years + 1 (year 0 = equity outflow)."""
        r = sim({"holding_years": 10})
        assert len(r.cashflow_series) == 11

    def test_year0_is_negative_equity(self):
        """First element of cashflow series = -equity_amount."""
        r = sim({"equity_amount": 120_000})
        assert r.cashflow_series[0] == -120_000

    def test_to_summary_dict_has_irr_key(self):
        r = sim()
        d = r.to_summary_dict()
        assert "irr" in d

    def test_cumulative_afa_grows_monotonically(self):
        r = sim()
        afa_vals = [s.cumulative_afa for s in r.snapshots]
        for i in range(len(afa_vals) - 1):
            assert afa_vals[i + 1] >= afa_vals[i]

    def test_loan_balance_decreases(self):
        """Loan balance should never increase year over year."""
        r = sim()
        balances = [s.loan_state.closing_balance for s in r.snapshots]
        for i in range(len(balances) - 1):
            assert balances[i + 1] <= balances[i] + 0.01  # float tolerance
