"""
tests/test_finance_engine.py
-----------------------------
Unit tests for finance_engine.py.

Covers:
  - Amortization precision (balance reaches 0, interest sums correctly)
  - Sondertilgung reduces balance faster
  - Refinancing switches rate at the right year
  - Annual cash flow flag logic
  - IRR sign-change requirements
  - Exit proceeds calculation
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math
import pytest
from finance_engine import (
    build_amortization_schedule,
    calc_purchase_costs,
    calc_equity_and_loan,
    calc_annual_cashflow,
    calc_exit_proceeds,
    build_cashflow_series,
    calc_irr,
    calc_npv,
)


# ─────────────────────────────────────────────────────────────────────────────
# Amortization schedule
# ─────────────────────────────────────────────────────────────────────────────

class TestAmortization:

    def test_fully_repaid_by_end(self):
        """Without Sondertilgung, loan should be repaid by holding_years."""
        s = build_amortization_schedule(200_000, 0.035, 20, 0.0)
        assert s.final_balance == pytest.approx(0.0, abs=1.0)

    def test_interest_decreases_over_time(self):
        """Interest payment should decrease monotonically (annuity)."""
        s = build_amortization_schedule(200_000, 0.035, 15, 0.0)
        interests = [st.annual_interest for st in s.yearly_states]
        for i in range(len(interests) - 1):
            if interests[i] > 0:
                assert interests[i] >= interests[i + 1]

    def test_sondertilgung_reduces_balance_faster(self):
        """With Sondertilgung, closing balance each year < without."""
        s_no   = build_amortization_schedule(300_000, 0.035, 15, 0.0)
        s_with = build_amortization_schedule(300_000, 0.035, 15, 0.02)
        for no, w in zip(s_no.yearly_states, s_with.yearly_states):
            assert w.closing_balance <= no.closing_balance

    def test_sondertilgung_saves_interest(self):
        """Total interest with Sondertilgung < without."""
        s_no   = build_amortization_schedule(300_000, 0.035, 15, 0.0)
        s_with = build_amortization_schedule(300_000, 0.035, 15, 0.02)
        assert s_with.total_interest_paid < s_no.total_interest_paid

    def test_sondertilgung_never_exceeds_balance(self):
        """Sondertilgung is capped at remaining balance — no negative balance."""
        s = build_amortization_schedule(50_000, 0.035, 15, 0.05)
        for st in s.yearly_states:
            assert st.closing_balance >= 0.0

    def test_schedule_length_matches_holding_years(self):
        """Schedule always has exactly holding_years entries."""
        for years in [5, 10, 15]:
            s = build_amortization_schedule(200_000, 0.035, years, 0.0)
            assert len(s.yearly_states) == years

    def test_refi_changes_rate_at_correct_year(self):
        """Refinancing at year 5 → rate changes exactly at year 5."""
        s = build_amortization_schedule(
            300_000, 0.04, 12, 0.0,
            refi_year=5, refi_rate=0.025
        )
        for st in s.yearly_states:
            if st.year_index < 5:
                assert st.interest_rate == pytest.approx(0.04, abs=0.001)
                assert st.is_refi_year is False
            elif st.year_index == 5:
                assert st.interest_rate == pytest.approx(0.025, abs=0.001)
                assert st.is_refi_year is True
            else:
                assert st.interest_rate == pytest.approx(0.025, abs=0.001)

    def test_refi_reduces_total_interest(self):
        """Refinancing to a lower rate reduces total interest paid."""
        s_no_refi = build_amortization_schedule(300_000, 0.04, 12, 0.0)
        s_refi    = build_amortization_schedule(
            300_000, 0.04, 12, 0.0, refi_year=5, refi_rate=0.025
        )
        assert s_refi.total_interest_paid < s_no_refi.total_interest_paid

    def test_zero_rate_loan(self):
        """Rate=0 should not raise; returns equal principal payments."""
        s = build_amortization_schedule(120_000, 0.0, 10, 0.0)
        for st in s.yearly_states:
            assert st.annual_interest == 0.0
        assert s.final_balance == pytest.approx(0.0, abs=1.0)

    def test_total_repaid_equals_principal(self):
        """Total principal repaid (regular + Sonder) == initial principal."""
        s = build_amortization_schedule(200_000, 0.035, 15, 0.02)
        assert s.total_principal_repaid == pytest.approx(200_000, abs=5.0)

    def test_invalid_principal_raises(self):
        with pytest.raises(ValueError):
            build_amortization_schedule(-1_000, 0.035, 10)

    def test_invalid_sondertilgung_raises(self):
        with pytest.raises(ValueError):
            build_amortization_schedule(200_000, 0.035, 10, sondertilgung_rate=0.15)


# ─────────────────────────────────────────────────────────────────────────────
# Purchase costs
# ─────────────────────────────────────────────────────────────────────────────

class TestPurchaseCosts:

    def test_total_nebenkosten_components_sum(self):
        """GrESt + Notar + Makler = total_nebenkosten."""
        pc = calc_purchase_costs(400_000, 0.06, 0.015, 0.0357)
        expected = pc["grunderwerbsteuer"] + pc["notar_grundbuch"] + pc["makler"]
        assert pc["total_nebenkosten"] == pytest.approx(expected, abs=1)

    def test_no_makler(self):
        """include_makler=False → makler=0."""
        pc = calc_purchase_costs(400_000, 0.06, include_makler=False)
        assert pc["makler"] == 0.0

    def test_total_investment_equals_price_plus_costs(self):
        pc = calc_purchase_costs(400_000, 0.06)
        assert pc["total_investment"] == pytest.approx(
            pc["purchase_price"] + pc["total_nebenkosten"], abs=1
        )


# ─────────────────────────────────────────────────────────────────────────────
# Equity and LTV
# ─────────────────────────────────────────────────────────────────────────────

class TestEquityLoan:

    def test_loan_plus_equity_equals_total(self):
        """equity + loan = total_needed."""
        pc  = calc_purchase_costs(400_000, 0.06)
        eq  = calc_equity_and_loan(400_000, pc["total_nebenkosten"], 100_000)
        assert eq["loan_amount"] + 100_000 == pytest.approx(eq["total_needed"], abs=1)

    def test_high_equity_low_loan(self):
        """Full cash purchase → loan = 0."""
        pc = calc_purchase_costs(400_000, 0.06)
        eq = calc_equity_and_loan(400_000, pc["total_nebenkosten"],
                                  equity_amount=500_000)
        assert eq["loan_amount"] == 0.0

    def test_ltv_calculation(self):
        """LTV = loan / purchase_price."""
        pc = calc_purchase_costs(400_000, 0.05)
        eq = calc_equity_and_loan(400_000, pc["total_nebenkosten"], 100_000)
        expected_ltv = eq["loan_amount"] / 400_000
        assert eq["ltv"] == pytest.approx(expected_ltv, rel=0.001)


# ─────────────────────────────────────────────────────────────────────────────
# Annual cash flow
# ─────────────────────────────────────────────────────────────────────────────

class TestAnnualCashflow:

    def test_positive_cashflow(self):
        cf = calc_annual_cashflow(20_000, 500, 10_000, 3_000, 0)
        assert cf["net_cashflow"] > 0
        assert cf["flag_negative_cashflow"] is False

    def test_negative_cashflow_flagged(self):
        cf = calc_annual_cashflow(10_000, 500, 15_000, 500, 0)
        assert cf["net_cashflow"] < 0
        assert cf["flag_negative_cashflow"] is True

    def test_components_sum_to_net(self):
        cf = calc_annual_cashflow(18_000, 1_000, 12_000, 2_500, 3_000)
        expected = (cf["operating_cashflow"]
                    + cf["financing_cashflow"]
                    + cf["tax_cashflow"])
        assert cf["net_cashflow"] == pytest.approx(expected, abs=0.01)

    def test_sondertilgung_is_cash_outflow(self):
        """Sondertilgung reduces net CF."""
        cf_no    = calc_annual_cashflow(18_000, 500, 10_000, 2_000, sondertilgung=0)
        cf_with  = calc_annual_cashflow(18_000, 500, 10_000, 2_000, sondertilgung=5_000)
        assert cf_with["net_cashflow"] < cf_no["net_cashflow"]


# ─────────────────────────────────────────────────────────────────────────────
# Exit proceeds
# ─────────────────────────────────────────────────────────────────────────────

class TestExitProceeds:

    def test_net_proceeds_less_than_gross(self):
        ep = calc_exit_proceeds(500_000, 200_000, speculation_tax=20_000)
        assert ep["net_proceeds"] < ep["gross_proceeds"]

    def test_no_speculation_tax(self):
        ep = calc_exit_proceeds(500_000, 200_000, speculation_tax=0)
        assert ep["net_proceeds"] == ep["gross_proceeds"]

    def test_selling_costs_deducted(self):
        ep = calc_exit_proceeds(500_000, 0, speculation_tax=0)
        assert ep["net_proceeds"] < 500_000
        assert ep["selling_costs"] > 0


# ─────────────────────────────────────────────────────────────────────────────
# IRR / NPV
# ─────────────────────────────────────────────────────────────────────────────

class TestIrrNpv:

    def test_irr_converges_for_typical_investment(self):
        """Typical real estate cashflow series → IRR converges."""
        series = build_cashflow_series(
            initial_equity=100_000,
            yearly_net_cashflows=[-2_000, -1_500, -1_000, 500, 1_000],
            exit_net_proceeds=120_000,
        )
        result = calc_irr(series)
        assert result["converged"] is True
        assert -0.5 < result["irr"] < 0.5   # sanity range

    def test_irr_undefined_all_negative(self):
        """All-negative cashflows → no sign change → IRR undefined."""
        series = [-100_000, -5_000, -5_000, -5_000]
        result = calc_irr(series)
        assert result["converged"] is False
        assert math.isnan(result["irr"])

    def test_npv_positive_at_zero_discount(self):
        """NPV at 0% discount = sum of cashflows."""
        series = [-100_000, 20_000, 20_000, 20_000, 80_000]
        result = calc_npv(series, 0.0)
        assert result["npv"] == pytest.approx(sum(series), abs=1)

    def test_npv_decreases_with_higher_discount(self):
        """Higher discount rate → lower NPV."""
        series = [-100_000, 30_000, 30_000, 30_000, 60_000]
        npv3 = calc_npv(series, 0.03)
        npv8 = calc_npv(series, 0.08)
        assert npv3["npv"] > npv8["npv"]

    def test_cashflow_series_length(self):
        """build_cashflow_series: length = 1 + len(yearly) ."""
        series = build_cashflow_series(50_000, [1_000, 2_000, 3_000], 80_000)
        assert len(series) == 4   # year 0 + 3 yearly

    def test_cashflow_series_year0_is_negative(self):
        """Year 0 = initial equity outflow (negative)."""
        series = build_cashflow_series(80_000, [1_000], 100_000)
        assert series[0] == -80_000

    def test_exit_added_to_last_year(self):
        """Exit proceeds are added to the last yearly cashflow."""
        series = build_cashflow_series(50_000, [1_000, 2_000], 60_000)
        assert series[-1] == pytest.approx(2_000 + 60_000, abs=1)
