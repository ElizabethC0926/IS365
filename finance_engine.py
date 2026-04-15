"""
finance_engine.py
-----------------
Financing and cash flow calculations for real estate investment simulation.

Design principles:
  - Pure functions: no global state, fully parallelisable.
  - Each function returns a dict (not a bare float) so simulator.py and
    Decision Log can trace every number back to its source.
  - All monetary results rounded to 2 decimal places (cent precision).
  - No tax logic here. Tax is handled exclusively by tax_engine.py.
    This module only produces the raw financial flows that tax_engine consumes.

Covered calculations:
  - Annuity loan amortization schedule (Annuitätendarlehen)
  - Sondertilgung (extra principal repayment)
  - Refinancing (Anschlussfinanzierung) at a new rate
  - Annual cash flow net of loan payments
  - Purchase cost breakdown (Kaufnebenkosten)
  - Equity / LTV calculations
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional
import math


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class YearlyLoanState:
    """Snapshot of the loan at the end of one calendar year."""
    year: int                     # Simulation year (calendar year)
    year_index: int               # Years since purchase (1-based)
    opening_balance: float        # Loan balance at start of year
    annual_interest: float        # Total interest paid during the year
    annual_principal: float       # Regular principal repaid during the year
    sondertilgung: float          # Extra principal repaid this year
    closing_balance: float        # Loan balance at end of year
    annual_payment: float         # Total cash out for loan (interest + principal + Sonder)
    interest_rate: float          # Rate in effect this year (may change at refi)
    is_refi_year: bool            # True if refinancing happened this year

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AmortizationSchedule:
    """Full loan schedule from purchase to end of holding period."""
    principal: float
    initial_rate: float
    holding_years: int
    sondertilgung_rate: float
    refi_year: Optional[int]
    refi_rate: Optional[float]
    yearly_states: list[YearlyLoanState] = field(default_factory=list)

    @property
    def total_interest_paid(self) -> float:
        return round(sum(s.annual_interest for s in self.yearly_states), 2)

    @property
    def total_principal_repaid(self) -> float:
        return round(sum(s.annual_principal + s.sondertilgung
                         for s in self.yearly_states), 2)

    @property
    def final_balance(self) -> float:
        if not self.yearly_states:
            return self.principal
        return self.yearly_states[-1].closing_balance

    def to_dict(self) -> dict:
        return {
            "principal": self.principal,
            "initial_rate": self.initial_rate,
            "holding_years": self.holding_years,
            "sondertilgung_rate": self.sondertilgung_rate,
            "refi_year": self.refi_year,
            "refi_rate": self.refi_rate,
            "total_interest_paid": self.total_interest_paid,
            "total_principal_repaid": self.total_principal_repaid,
            "final_balance": self.final_balance,
            "yearly_states": [s.to_dict() for s in self.yearly_states],
        }


# ---------------------------------------------------------------------------
# Core loan calculations
# ---------------------------------------------------------------------------

def _annuity_payment(principal: float, annual_rate: float, years: int) -> float:
    """
    Standard annuity formula: fixed annual payment that fully amortises
    the loan over `years` at `annual_rate`.

    Edge case: rate == 0 -> equal principal payments with no interest.
    """
    if annual_rate <= 0:
        return round(principal / years, 2) if years > 0 else 0.0
    r = annual_rate
    return round(principal * r * (1 + r) ** years / ((1 + r) ** years - 1), 2)


def build_amortization_schedule(
    principal: float,
    annual_rate: float,
    holding_years: int,
    sondertilgung_rate: float = 0.0,
    refi_year: Optional[int] = None,
    refi_rate: Optional[float] = None,
    purchase_year: int = 2025,
) -> AmortizationSchedule:
    """
    Build a full year-by-year loan amortization schedule.

    Parameters
    ----------
    principal : float
        Initial loan amount (Darlehensbetrag).
    annual_rate : float
        Initial annual interest rate as a decimal (e.g. 0.035 for 3.5%).
    holding_years : int
        Number of years the property is held (= length of the schedule).
    sondertilgung_rate : float
        Extra annual principal repayment as a fraction of the ORIGINAL
        principal (e.g. 0.02 = 2% of original loan per year).
        Applied at the end of each year. Max typically 5% in German contracts.
    refi_year : int, optional
        Year index (1-based) at which refinancing occurs.
        The annuity is recalculated from this point at refi_rate.
    refi_rate : float, optional
        New annual rate after refinancing. Required if refi_year is set.
    purchase_year : int
        Calendar year of purchase (used to label YearlyLoanState.year).

    Returns
    -------
    AmortizationSchedule
        Full schedule with one YearlyLoanState per year.

    Notes
    -----
    German Annuitätendarlehen: the annuity (Annuität) is fixed for the
    Zinsbindungsfrist (interest rate lock period). After that, the remaining
    balance is refinanced. Sondertilgung reduces the balance but does NOT
    change the regular annuity payment (it shortens the effective term).
    """
    if principal <= 0:
        raise ValueError(f"Principal must be positive, got {principal}")
    if annual_rate < 0:
        raise ValueError(f"Annual rate must be >= 0, got {annual_rate}")
    if holding_years <= 0:
        raise ValueError(f"holding_years must be positive, got {holding_years}")
    if sondertilgung_rate < 0 or sondertilgung_rate > 0.10:
        raise ValueError(f"sondertilgung_rate {sondertilgung_rate} out of range [0, 0.10]")
    if refi_year is not None and refi_rate is None:
        raise ValueError("refi_rate must be provided when refi_year is set")

    schedule = AmortizationSchedule(
        principal=principal,
        initial_rate=annual_rate,
        holding_years=holding_years,
        sondertilgung_rate=sondertilgung_rate,
        refi_year=refi_year,
        refi_rate=refi_rate,
    )

    balance = principal
    current_rate = annual_rate
    # Remaining term for annuity calculation — recalculated at refi point
    remaining_term = holding_years
    annuity = _annuity_payment(balance, current_rate, remaining_term)
    sonder_amount = round(principal * sondertilgung_rate, 2)

    for idx in range(1, holding_years + 1):
        calendar_year = purchase_year + idx
        is_refi = False

        # Refinancing: recalculate annuity from new rate and remaining balance
        if refi_year is not None and idx == refi_year and balance > 0:
            current_rate = refi_rate
            remaining_term = holding_years - idx + 1
            annuity = _annuity_payment(balance, current_rate, remaining_term)
            is_refi = True

        opening = round(balance, 2)

        # Interest accrues on opening balance
        interest = round(opening * current_rate, 2)

        # Regular principal = annuity minus interest (floored at 0 if loan nearly paid)
        regular_principal = max(0.0, round(annuity - interest, 2))
        # Clamp so we never repay more than the balance
        regular_principal = min(regular_principal, opening)

        balance_after_regular = round(opening - regular_principal, 2)

        # Sondertilgung: capped at remaining balance
        sonder = min(sonder_amount, balance_after_regular)
        sonder = round(sonder, 2)

        closing = round(balance_after_regular - sonder, 2)
        closing = max(0.0, closing)   # prevent floating point negatives

        total_payment = round(interest + regular_principal + sonder, 2)

        state = YearlyLoanState(
            year=calendar_year,
            year_index=idx,
            opening_balance=opening,
            annual_interest=interest,
            annual_principal=regular_principal,
            sondertilgung=sonder,
            closing_balance=closing,
            annual_payment=total_payment,
            interest_rate=current_rate,
            is_refi_year=is_refi,
        )
        schedule.yearly_states.append(state)
        balance = closing

        # Loan fully repaid early — fill remaining years with zeros
        if balance <= 0:
            for remaining_idx in range(idx + 1, holding_years + 1):
                schedule.yearly_states.append(YearlyLoanState(
                    year=purchase_year + remaining_idx,
                    year_index=remaining_idx,
                    opening_balance=0.0,
                    annual_interest=0.0,
                    annual_principal=0.0,
                    sondertilgung=0.0,
                    closing_balance=0.0,
                    annual_payment=0.0,
                    interest_rate=current_rate,
                    is_refi_year=False,
                ))
            break

    return schedule


# ---------------------------------------------------------------------------
# Purchase cost breakdown
# ---------------------------------------------------------------------------

def calc_purchase_costs(
    purchase_price: float,
    grunderwerbsteuer_rate: float,
    notar_rate: float = 0.015,
    makler_rate: float = 0.0357,
    include_makler: bool = True,
) -> dict:
    """
    Calculate all one-time purchase costs (Kaufnebenkosten).

    These costs:
      - Are paid at purchase and not recovered.
      - Increase the AfA base (Anschaffungskosten) for building depreciation.
      - Are NOT deductible as Werbungskosten in the year of purchase.

    Parameters
    ----------
    purchase_price : float
        Agreed purchase price (Kaufpreis).
    grunderwerbsteuer_rate : float
        State-specific land transfer tax rate (from tax_engine).
    notar_rate : float
        Notary + land registry fee rate. Typically ~1.5%.
    makler_rate : float
        Buyer's share of broker commission. Typically 3.57% (incl. VAT).
    include_makler : bool
        Set False for off-market purchases or when seller pays full commission.
    """
    grunderwerbsteuer = round(purchase_price * grunderwerbsteuer_rate, 2)
    notar             = round(purchase_price * notar_rate, 2)
    makler            = round(purchase_price * makler_rate, 2) if include_makler else 0.0
    total_nebenkosten = round(grunderwerbsteuer + notar + makler, 2)
    total_investment  = round(purchase_price + total_nebenkosten, 2)

    return {
        "purchase_price": purchase_price,
        "grunderwerbsteuer": grunderwerbsteuer,
        "grunderwerbsteuer_rate": grunderwerbsteuer_rate,
        "notar_grundbuch": notar,
        "notar_rate": notar_rate,
        "makler": makler,
        "makler_rate": makler_rate if include_makler else 0.0,
        "total_nebenkosten": total_nebenkosten,
        "total_investment": total_investment,
        "nebenkosten_rate": round(total_nebenkosten / purchase_price, 4),
    }


# ---------------------------------------------------------------------------
# Equity and LTV
# ---------------------------------------------------------------------------

def calc_equity_and_loan(
    purchase_price: float,
    total_nebenkosten: float,
    equity_amount: float,
) -> dict:
    """
    Derive loan amount and LTV from equity contribution.

    German convention: LTV is calculated against purchase price only,
    not total investment (nebenkosten are typically equity-funded).

    Parameters
    ----------
    equity_amount : float
        Cash equity the buyer brings. Must cover at least the Nebenkosten
        for a standard loan (banks rarely finance Nebenkosten).
    """
    total_needed  = round(purchase_price + total_nebenkosten, 2)
    loan_amount   = round(max(0.0, total_needed - equity_amount), 2)
    ltv           = round(loan_amount / purchase_price, 4) if purchase_price > 0 else 0.0
    equity_ratio  = round(equity_amount / total_needed, 4) if total_needed > 0 else 0.0

    return {
        "purchase_price": purchase_price,
        "total_nebenkosten": total_nebenkosten,
        "total_needed": total_needed,
        "equity_amount": equity_amount,
        "loan_amount": loan_amount,
        "ltv": ltv,
        "equity_ratio": equity_ratio,
        "nebenkosten_covered_by_equity": equity_amount >= total_nebenkosten,
    }


# ---------------------------------------------------------------------------
# Annual cash flow
# ---------------------------------------------------------------------------

def calc_annual_cashflow(
    rental_income_gross: float,
    non_deductible_costs: float,
    loan_payment: float,
    tax_refund: float,
    sondertilgung: float = 0.0,
) -> dict:
    """
    Net annual cash flow from the investor's bank account perspective.

    Formula:
        CF = rental_income_gross
             - non_deductible_costs   (Hausgeld, Instandhaltungsrücklage etc.)
             - loan_payment           (interest + regular principal)
             - sondertilgung          (extra repayment — cash out, not a cost)
             + tax_refund             (Steuererstattung from negative VuV income)

    Note: principal repayment is NOT a cost (it builds equity) but IS a cash
    outflow. We show it separately so the caller can distinguish
    "economic return" from "accounting cash flow".

    Parameters
    ----------
    rental_income_gross : float
        Annual gross rent received (Bruttomieteinnahmen).
    non_deductible_costs : float
        Running costs NOT covered by the loan payment and NOT tax-deductible
        (e.g. portion of Hausgeld that is Instandhaltungsrücklage).
    loan_payment : float
        Total annual loan cash outflow (interest + regular principal).
        From YearlyLoanState.annual_payment minus sondertilgung.
    tax_refund : float
        Tax saved / refund received due to negative VuV income.
        Positive value = money back from Finanzamt.
        Computed by tax_engine, passed in here.
    sondertilgung : float
        Extra principal payment this year (cash out, equity builder).
    """
    operating_cf = round(rental_income_gross - non_deductible_costs, 2)
    financing_cf = round(-loan_payment - sondertilgung, 2)
    tax_cf       = round(tax_refund, 2)
    net_cf       = round(operating_cf + financing_cf + tax_cf, 2)

    return {
        "rental_income_gross": rental_income_gross,
        "non_deductible_costs": non_deductible_costs,
        "operating_cashflow": operating_cf,
        "loan_payment": loan_payment,
        "sondertilgung": sondertilgung,
        "financing_cashflow": financing_cf,
        "tax_refund": tax_refund,
        "tax_cashflow": tax_cf,
        "net_cashflow": net_cf,
        "flag_negative_cashflow": net_cf < 0,
    }


# ---------------------------------------------------------------------------
# Exit proceeds
# ---------------------------------------------------------------------------

def calc_exit_proceeds(
    sale_price: float,
    remaining_loan_balance: float,
    makler_sell_rate: float = 0.0357,
    notar_sell_rate: float = 0.010,
    speculation_tax: float = 0.0,
) -> dict:
    """
    Net proceeds to the investor after selling the property.

    Formula:
        Net proceeds = sale_price
                       - remaining_loan_balance   (repay bank)
                       - selling_costs            (Makler + Notar)
                       - speculation_tax          (§23 EStG, if < 10 years)

    Parameters
    ----------
    sale_price : float
        Agreed sale price.
    remaining_loan_balance : float
        Outstanding loan balance at exit (from amortization schedule).
    makler_sell_rate : float
        Seller's broker commission (typically 3.57% incl. VAT).
    notar_sell_rate : float
        Notary costs on sale side (typically ~1%).
    speculation_tax : float
        Tax owed on gain under §23 EStG (0 if held >= 10 years).
    """
    selling_costs   = round(sale_price * (makler_sell_rate + notar_sell_rate), 2)
    gross_proceeds  = round(sale_price - remaining_loan_balance - selling_costs, 2)
    net_proceeds    = round(gross_proceeds - speculation_tax, 2)

    return {
        "sale_price": sale_price,
        "remaining_loan_balance": remaining_loan_balance,
        "selling_costs": selling_costs,
        "makler_sell_rate": makler_sell_rate,
        "notar_sell_rate": notar_sell_rate,
        "gross_proceeds": gross_proceeds,
        "speculation_tax": speculation_tax,
        "net_proceeds": net_proceeds,
    }


# ---------------------------------------------------------------------------
# IRR / NPV helpers
# ---------------------------------------------------------------------------

def build_cashflow_series(
    initial_equity: float,
    yearly_net_cashflows: list[float],
    exit_net_proceeds: float,
) -> list[float]:
    """
    Assemble the investor's complete cash flow series for IRR calculation.

    Year 0  : -initial_equity  (money paid out at purchase)
    Year 1..N: net_cashflow per year (can be negative)
    Year N  : exit_net_proceeds added to year N cash flow

    Returns a list where index 0 = year 0 (purchase), index N = exit year.
    """
    if not yearly_net_cashflows:
        raise ValueError("yearly_net_cashflows must not be empty")

    series = [-abs(initial_equity)]
    for i, cf in enumerate(yearly_net_cashflows):
        if i == len(yearly_net_cashflows) - 1:
            series.append(round(cf + exit_net_proceeds, 2))
        else:
            series.append(round(cf, 2))
    return series


def calc_irr(cashflows: list[float], max_iterations: int = 1000) -> dict:
    """
    Calculate Internal Rate of Return using Newton-Raphson iteration.

    Requires at least one sign change in the cash flow series
    (typical: negative initial investment, positive later flows).

    Returns nan if no solution converges (e.g. all-negative cash flows).
    """
    if len(cashflows) < 2:
        return {"irr": float("nan"), "converged": False,
                "reason": "Need at least 2 cash flows"}

    has_negative = any(cf < 0 for cf in cashflows)
    has_positive = any(cf > 0 for cf in cashflows)
    if not (has_negative and has_positive):
        return {"irr": float("nan"), "converged": False,
                "reason": "No sign change in cash flows — IRR undefined"}

    def npv_and_derivative(rate: float):
        npv  = 0.0
        dnpv = 0.0
        for t, cf in enumerate(cashflows):
            discount = (1 + rate) ** t
            npv  += cf / discount
            dnpv -= t * cf / ((1 + rate) ** (t + 1))
        return npv, dnpv

    # Try multiple starting points to avoid local minima
    for guess in [0.05, 0.10, 0.15, -0.05, 0.20, 0.01]:
        rate = guess
        for _ in range(max_iterations):
            npv, deriv = npv_and_derivative(rate)
            if abs(deriv) < 1e-12:
                break
            new_rate = rate - npv / deriv
            if abs(new_rate - rate) < 1e-8:
                return {
                    "irr": round(new_rate, 6),
                    "converged": True,
                    "reason": "Newton-Raphson converged",
                }
            rate = new_rate

    return {"irr": float("nan"), "converged": False,
            "reason": "Did not converge after multiple starting points"}


def calc_npv(cashflows: list[float], discount_rate: float) -> dict:
    """Standard NPV at a given discount rate."""
    if discount_rate <= -1:
        return {"npv": float("nan"), "reason": "Discount rate must be > -1"}
    npv = sum(cf / (1 + discount_rate) ** t for t, cf in enumerate(cashflows))
    return {
        "npv": round(npv, 2),
        "discount_rate": discount_rate,
        "num_periods": len(cashflows) - 1,
    }
