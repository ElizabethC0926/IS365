"""
simulator.py
------------
Single-case simulation engine.

Takes a parameter dict, builds a PropertyCase, drives it year by year,
detects FLAGS, computes IRR, and returns a complete SimulationResult.

Design:
  - Pure function: run_simulation() has no side effects or global state.
  - Fully parallelisable: safe to call from multiprocessing.Pool.
  - All numbers traceable: SimulationResult contains the full yearly history.
  - sale_price is passed explicitly so the caller (sampler / user) controls
    the assumed appreciation. Default = purchase price (no appreciation).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
from property_model import PropertyCase, YearlySnapshot, ExitResult
from finance_engine import build_cashflow_series, calc_irr, calc_npv


# ---------------------------------------------------------------------------
# FLAG detection
# ---------------------------------------------------------------------------

def detect_flags(
    snapshots: list[YearlySnapshot],
    exit_result: ExitResult,
    personal_income: float,
) -> dict:
    """
    Evaluate all four FLAGS across the full simulation history.

    Returns a dict with one entry per FLAG:
        {
          "FLAG_NAME": {
              "triggered": bool,
              "year_index": int | None,   # first year it triggered
              "detail": str               # human-readable explanation
          }
        }
    """
    flags = {}

    # ---- FLAG_15_PERCENT_HIT ----
    hit_15 = next(
        (s for s in snapshots if s.rule_15pct["triggered"]), None
    )
    flags["FLAG_15_PERCENT_HIT"] = {
        "triggered": hit_15 is not None,
        "year_index": hit_15.year_index if hit_15 else None,
        "detail": (
            f"Yr{hit_15.year_index}: renovation €{hit_15.renovation_this_year:,.0f} "
            f"pushed cumulative spend above 15% limit "
            f"(€{hit_15.rule_15pct['limit_amount']:,.0f}). "
            f"Costs capitalised, not immediately deductible."
            if hit_15 else "15% rule not triggered."
        ),
    }

    # ---- FLAG_RENT_TOO_LOW ----
    low_rent = next(
        (s for s in snapshots if s.rent_rule["flag_rent_too_low"]), None
    )
    flags["FLAG_RENT_TOO_LOW"] = {
        "triggered": low_rent is not None,
        "year_index": low_rent.year_index if low_rent else None,
        "detail": (
            f"Yr{low_rent.year_index}: rent ratio {low_rent.rent_rule['rent_ratio']:.0%} "
            f"below 66% threshold → zone='{low_rent.rent_rule['zone']}', "
            f"deduction ratio={low_rent.rent_rule['deduction_ratio']}."
            if low_rent else "Rent is at or above 66% of market rate."
        ),
    }

    # ---- FLAG_TAX_WASTE ----
    # Tax waste: VuV loss exceeds personal income in a year,
    # so excess deductions cannot be used even with Verlustausgleich.
    tax_waste_year = None
    for s in snapshots:
        if s.verpachtung_income < 0:
            loss = abs(s.verpachtung_income)
            if loss > personal_income:
                tax_waste_year = s
                break
    flags["FLAG_TAX_WASTE"] = {
        "triggered": tax_waste_year is not None,
        "year_index": tax_waste_year.year_index if tax_waste_year else None,
        "detail": (
            f"Yr{tax_waste_year.year_index}: VuV loss "
            f"€{abs(tax_waste_year.verpachtung_income):,.0f} exceeds "
            f"personal income €{personal_income:,.0f}. "
            f"Excess deductions wasted (no Verlustvortrag modelled)."
            if tax_waste_year else "No tax deduction waste detected."
        ),
    }

    # ---- FLAG_NEGATIVE_CASHFLOW ----
    neg_cf = next(
        (s for s in snapshots if s.cashflow["flag_negative_cashflow"]), None
    )
    flags["FLAG_NEGATIVE_CASHFLOW"] = {
        "triggered": neg_cf is not None,
        "year_index": neg_cf.year_index if neg_cf else None,
        "detail": (
            f"Yr{neg_cf.year_index}: net cashflow "
            f"€{neg_cf.cashflow['net_cashflow']:,.0f} (negative)."
            if neg_cf else "Cash flow positive in all years."
        ),
    }

    return flags


# ---------------------------------------------------------------------------
# SimulationResult
# ---------------------------------------------------------------------------

@dataclass
class SimulationResult:
    """
    Complete output of one strategy simulation.

    This is the unit that output_formatter.py converts into
    one row of the results table.
    """
    # Input echo (for traceability)
    params: dict

    # Year-by-year history
    snapshots: list[YearlySnapshot]

    # Exit
    exit_result: ExitResult

    # FLAGS (4 standard flags)
    flags: dict

    # Return metrics
    irr_result: dict          # from finance_engine.calc_irr
    npv_result: dict          # from finance_engine.calc_npv
    cashflow_series: list[float]  # year-0 to year-N series fed into IRR

    # Summary convenience fields (pre-extracted for output_formatter)
    total_tax_saved: float    # Sum of tax_delta across all years
    total_interest_paid: float
    total_net_cashflow: float # Sum of annual net cashflows (excl. exit)

    def irr(self) -> float:
        return self.irr_result.get("irr", float("nan"))

    def npv(self) -> float:
        return self.npv_result.get("npv", float("nan"))

    def any_flag(self) -> bool:
        return any(v["triggered"] for v in self.flags.values())

    def to_summary_dict(self) -> dict:
        """
        Flat dict for output_formatter → one row in the results table.
        Contains all input params + key output metrics + FLAGS.
        """
        d = dict(self.params)  # all input dimensions as columns

        # Return metrics
        d["irr"]              = self.irr()
        d["npv"]              = self.npv()
        d["irr_converged"]    = self.irr_result.get("converged", False)

        # Exit
        er = self.exit_result.to_dict()
        d.update(er)

        # Aggregates
        d["total_tax_saved"]     = self.total_tax_saved
        d["total_interest_paid"] = self.total_interest_paid
        d["total_net_cashflow"]  = self.total_net_cashflow

        # FLAGS — one column per flag (bool) + one for first trigger year
        for flag_name, flag_data in self.flags.items():
            d[flag_name]                    = flag_data["triggered"]
            d[f"{flag_name}_year"]          = flag_data["year_index"]
            d[f"{flag_name}_detail"]        = flag_data["detail"]

        # Any flag triggered
        d["any_flag"] = self.any_flag()

        return d


# ---------------------------------------------------------------------------
# Main simulation function
# ---------------------------------------------------------------------------

def run_simulation(
    params: dict,
    sale_price: Optional[float] = None,
    npv_discount_rate: float = 0.03,
    tax_params_path: str = "tax_params.json",
) -> SimulationResult:
    """
    Run a complete simulation for one strategy parameter set.

    Parameters
    ----------
    params : dict
        All fields required by PropertyCase. See PropertyCase for the full list.
    sale_price : float, optional
        Assumed sale price at exit. Defaults to purchase_price (no appreciation).
        Pass a value > purchase_price to model capital appreciation.
    npv_discount_rate : float
        Discount rate for NPV calculation. Default 3% (typical risk-free proxy).
    tax_params_path : str
        Path to tax_params.json (forwarded to TaxEngine).

    Returns
    -------
    SimulationResult
        Complete simulation output, ready for output_formatter.py.
    """
    from tax_engine import TaxEngine

    # Build the case
    tax_engine = TaxEngine(tax_params_path)
    case = PropertyCase(**params, _tax_engine=tax_engine)

    if sale_price is None:
        sale_price = case.purchase_price   # neutral: no appreciation

    # ---- Run year by year ----
    snapshots: list[YearlySnapshot] = []
    for yr in range(1, case.holding_years + 1):
        snapshot = case.step_year(yr)
        snapshots.append(snapshot)

    # ---- Exit ----
    # Override the placeholder sale_price in calc_exit
    exit_result = case.calc_exit(case.holding_years)
    # Patch with the actual sale price (calc_exit uses purchase_price as placeholder)
    from finance_engine import calc_exit_proceeds
    spec_tax = tax_engine.calc_speculation_tax(
        sale_price=sale_price,
        original_purchase_price=case.purchase_price,
        cumulative_afa_claimed=case._cumulative_afa,
        holding_years=case.holding_years,
        annual_income_in_exit_year=case.personal_income,
        filing_status=case.filing_status,
        simulation_year=case.purchase_year + case.holding_years,
    )
    exit_proceeds = calc_exit_proceeds(
        sale_price=sale_price,
        remaining_loan_balance=case._schedule.final_balance,
        speculation_tax=spec_tax["speculation_tax"],
    )
    from property_model import ExitResult as ER
    exit_result = ER(
        year_index=case.holding_years,
        calendar_year=case.purchase_year + case.holding_years,
        sale_price=sale_price,
        remaining_loan_balance=case._schedule.final_balance,
        cumulative_afa_at_exit=case._cumulative_afa,
        speculation_tax_result=spec_tax,
        exit_proceeds=exit_proceeds,
    )

    # ---- FLAGS ----
    flags = detect_flags(snapshots, exit_result, case.personal_income)

    # ---- IRR / NPV ----
    annual_net_cfs = [s.cashflow["net_cashflow"] for s in snapshots]
    cf_series = build_cashflow_series(
        initial_equity=case.equity_amount,
        yearly_net_cashflows=annual_net_cfs,
        exit_net_proceeds=exit_result.exit_proceeds["net_proceeds"],
    )
    irr_result = calc_irr(cf_series)
    npv_result = calc_npv(cf_series, npv_discount_rate)

    # ---- Aggregates ----
    total_tax_saved     = round(sum(s.tax_delta for s in snapshots), 2)
    total_interest_paid = round(sum(s.loan_state.annual_interest for s in snapshots), 2)
    total_net_cashflow  = round(sum(s.cashflow["net_cashflow"] for s in snapshots), 2)

    return SimulationResult(
        params=params,
        snapshots=snapshots,
        exit_result=exit_result,
        flags=flags,
        irr_result=irr_result,
        npv_result=npv_result,
        cashflow_series=cf_series,
        total_tax_saved=total_tax_saved,
        total_interest_paid=total_interest_paid,
        total_net_cashflow=total_net_cashflow,
    )
