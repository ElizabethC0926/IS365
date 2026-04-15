"""
property_model.py
-----------------
Core data model for a single investment strategy (one row in the results table).

A PropertyCase holds ALL parameters that define one specific combination of:
  - What the property is (price, location, building type)
  - How it is financed (LTV, rate, Sondertilgung, refi)
  - How it is used (rental ratio, partial own-use)
  - How it is managed (renovation timing, asset split, land ratio)
  - When it exits (holding years)

PropertyCase does NOT contain any calculation logic.
It delegates everything to tax_engine and finance_engine.

The simulator calls step_year(year_index) once per year in sequence,
building up a list of YearlySnapshot objects that represent the
complete financial history of this strategy.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Literal
from tax_engine import TaxEngine
from finance_engine import (
    build_amortization_schedule,
    calc_purchase_costs,
    calc_equity_and_loan,
    AmortizationSchedule,
)


# ---------------------------------------------------------------------------
# Input parameter dataclass
# ---------------------------------------------------------------------------

@dataclass
class PropertyCase:
    """
    All parameters defining one investment strategy.

    Dimensions map directly to V1.0 spec:
      Core Strategic  : usage, holding_years, rental_ratio
      Tax Tactical    : renovation_year, renovation_amount, asset_split, land_ratio
      Financial       : equity_amount, annual_rate, sondertilgung_rate, refi_year, refi_rate

    Fixed inputs (same for all strategies for a given user):
      purchase_price, state, purchase_year, personal_income, filing_status,
      market_rent_annual, building_type
    """

    # ---- Fixed inputs (user-provided, same across all strategies) ----
    purchase_price: float           # Kaufpreis in EUR
    state: str                      # German Bundesland (for Grunderwerbsteuer)
    purchase_year: int              # Calendar year of purchase
    personal_income: float          # Annual employment/other income (for tax calc)
    filing_status: str              # "single" | "married"
    market_rent_annual: float       # Ortsübliche Jahresmiete (benchmark)
    building_type: str              # "standard" | "neubau_post_2023" | "denkmal"

    # ---- Core Strategic (V1.0 Dimension 1) ----
    usage: Literal["full_rental", "partial", "own_use"]
    holding_years: int              # 1–15
    rental_ratio: float             # Actual rent / market rent (e.g. 0.9 = 90%)

    # ---- Tax Tactical (V1.0 Dimension 2) ----
    renovation_year: int            # Year index (1-based) of major renovation
    renovation_amount: float        # EUR spent on renovation in renovation_year
    asset_split: float              # EUR value of movables (Küche etc.) in contract
    land_ratio: float               # Fraction of purchase price = Grundanteil (0.1–0.5)

    # ---- Financial Leverage (V1.0 Dimension 3) ----
    equity_amount: float            # EUR equity brought by investor
    annual_rate: float              # Initial loan interest rate (decimal)
    sondertilgung_rate: float       # Extra repayment as fraction of original loan
    refi_year: Optional[int]        # Year index for refinancing (None = no refi)
    refi_rate: Optional[float]      # Rate after refinancing

    # ---- Derived at init (not user inputs) ----
    _tax_engine: TaxEngine = field(default=None, repr=False, compare=False)
    _schedule: AmortizationSchedule = field(default=None, repr=False, compare=False)
    _purchase_costs: dict = field(default=None, repr=False, compare=False)
    _equity_loan: dict = field(default=None, repr=False, compare=False)
    _cumulative_renovation: float = field(default=0.0, repr=False, compare=False)
    _cumulative_afa: float = field(default=0.0, repr=False, compare=False)

    def __post_init__(self):
        """Pre-compute derived values that are fixed for the lifetime of this case."""
        if self._tax_engine is None:
            self._tax_engine = TaxEngine()

        # Purchase cost breakdown
        grest_rate = self._tax_engine.calc_grunderwerbsteuer(
            self.purchase_price, self.state, self.purchase_year
        )["rate"]
        self._purchase_costs = calc_purchase_costs(
            self.purchase_price, grunderwerbsteuer_rate=grest_rate
        )

        # Equity and loan
        self._equity_loan = calc_equity_and_loan(
            self.purchase_price,
            self._purchase_costs["total_nebenkosten"],
            self.equity_amount,
        )

        # Build full amortization schedule upfront
        self._schedule = build_amortization_schedule(
            principal=self._equity_loan["loan_amount"],
            annual_rate=self.annual_rate,
            holding_years=self.holding_years,
            sondertilgung_rate=self.sondertilgung_rate,
            refi_year=self.refi_year,
            refi_rate=self.refi_rate,
            purchase_year=self.purchase_year,
        )

    # ------------------------------------------------------------------ #
    #  Derived property values
    # ------------------------------------------------------------------ #

    @property
    def building_value(self) -> float:
        """Purchase price allocated to the building (excl. land and movables)."""
        land_value   = round(self.purchase_price * self.land_ratio, 2)
        movable_value = self.asset_split
        return round(self.purchase_price - land_value - movable_value, 2)

    @property
    def actual_rent_annual(self) -> float:
        """Actual annual rent received, based on rental_ratio × market rent."""
        if self.usage == "own_use":
            return 0.0
        return round(self.market_rent_annual * self.rental_ratio, 2)

    @property
    def allocation_ratio(self) -> float:
        """
        Fraction of costs allocatable to rental use.
        full_rental -> 1.0, partial -> 0.5, own_use -> 0.0
        """
        return {"full_rental": 1.0, "partial": 0.5, "own_use": 0.0}[self.usage]

    @property
    def loan_amount(self) -> float:
        return self._equity_loan["loan_amount"]

    @property
    def total_investment(self) -> float:
        return self._purchase_costs["total_investment"]

    # ------------------------------------------------------------------ #
    #  Year-by-year simulation step
    # ------------------------------------------------------------------ #

    def step_year(self, year_index: int) -> "YearlySnapshot":
        """
        Compute the complete financial picture for simulation year `year_index`.

        year_index is 1-based (year 1 = first full year of ownership).
        Calendar year = purchase_year + year_index.

        This method is stateful only in that it accumulates:
          - _cumulative_renovation (for the 15% rule check)
          - _cumulative_afa       (for the §23 speculation tax base at exit)

        It must be called in order (year 1, 2, 3, …).
        """
        if year_index < 1 or year_index > self.holding_years:
            raise ValueError(
                f"year_index {year_index} out of range [1, {self.holding_years}]"
            )

        calendar_year = self.purchase_year + year_index
        tax = self._tax_engine
        loan_state = self._schedule.yearly_states[year_index - 1]

        # ---- Renovation ----
        renovation_this_year = (
            self.renovation_amount if year_index == self.renovation_year else 0.0
        )
        self._cumulative_renovation = round(
            self._cumulative_renovation + renovation_this_year, 2
        )

        # ---- 15% rule check ----
        rule_15pct = tax.check_15pct_rule(
            renovation_cumulative_net=self._cumulative_renovation,
            purchase_price_net=self.purchase_price,
            years_since_purchase=year_index,
            simulation_year=calendar_year,
        )

        # Renovation is only immediately deductible if 15% rule NOT triggered
        renovation_deductible = (
            0.0 if rule_15pct["triggered"] else renovation_this_year
        )
        # If triggered, it gets capitalised → added to building AfA base
        renovation_capitalised = (
            renovation_this_year if rule_15pct["triggered"] else 0.0
        )

        # ---- AfA ----
        # Building value for AfA: original building value + any capitalised renovation
        afa_building_base = self.building_value + renovation_capitalised
        afa_result = tax.calc_afa(
            building_value=afa_building_base,
            movable_value=self.asset_split,
            year_of_purchase=self.purchase_year,
            simulation_year=calendar_year,
            building_type=self.building_type,
        )
        self._cumulative_afa = round(
            self._cumulative_afa + afa_result["total_afa"], 2
        )

        # ---- Rent rule ----
        rent_rule = tax.check_rent_rule(
            actual_rent_annual=self.actual_rent_annual,
            market_rent_annual=self.market_rent_annual,
            simulation_year=calendar_year,
        )
        deduction_ratio = rent_rule["deduction_ratio"] * self.allocation_ratio

        # ---- Werbungskosten ----
        werbungskosten = tax.calc_werbungskosten(
            interest_paid=loan_state.annual_interest,
            afa_total=afa_result["total_afa"],
            renovation_deductible=renovation_deductible,
            management_costs=self.actual_rent_annual * 0.02,   # ~2% of rent
            insurance_costs=self.purchase_price * 0.001,       # ~0.1% of value
            other_costs=500.0,                                  # flat misc costs
            deduction_ratio=deduction_ratio,
            simulation_year=calendar_year,
        )

        # ---- Taxable VuV income ----
        verpachtung_income = round(
            self.actual_rent_annual - werbungskosten["total_deductible"], 2
        )
        # Negative VuV income = loss, offsets personal income (vertikaler Verlustausgleich)
        total_taxable_income = round(
            self.personal_income + verpachtung_income, 2
        )
        total_taxable_income = max(0.0, total_taxable_income)

        # ---- Income tax ----
        tax_with_property = tax.calc_income_tax(
            total_taxable_income, calendar_year, self.filing_status
        )
        tax_without_property = tax.calc_income_tax(
            self.personal_income, calendar_year, self.filing_status
        )
        # Tax refund (positive) or extra tax (negative) due to property
        tax_delta = round(
            tax_without_property["total_tax"] - tax_with_property["total_tax"], 2
        )

        # ---- Annual cash flow ----
        from finance_engine import calc_annual_cashflow
        cashflow = calc_annual_cashflow(
            rental_income_gross=self.actual_rent_annual,
            non_deductible_costs=500.0,   # Instandhaltungsrücklage etc.
            loan_payment=loan_state.annual_payment - loan_state.sondertilgung,
            tax_refund=tax_delta,
            sondertilgung=loan_state.sondertilgung,
        )

        return YearlySnapshot(
            year_index=year_index,
            calendar_year=calendar_year,
            # Loan
            loan_state=loan_state,
            # Rental
            actual_rent_annual=self.actual_rent_annual,
            rent_rule=rent_rule,
            deduction_ratio=deduction_ratio,
            # Renovation
            renovation_this_year=renovation_this_year,
            renovation_deductible=renovation_deductible,
            renovation_capitalised=renovation_capitalised,
            rule_15pct=rule_15pct,
            cumulative_renovation=self._cumulative_renovation,
            # AfA
            afa_result=afa_result,
            cumulative_afa=self._cumulative_afa,
            # Tax
            werbungskosten=werbungskosten,
            verpachtung_income=verpachtung_income,
            total_taxable_income=total_taxable_income,
            tax_with_property=tax_with_property,
            tax_without_property=tax_without_property,
            tax_delta=tax_delta,
            # Cash flow
            cashflow=cashflow,
        )

    def calc_exit(self, year_index: int) -> "ExitResult":
        """
        Compute exit (sale) proceeds at the end of year_index.

        Includes §23 speculation tax if held < 10 years.
        Must be called after all step_year() calls up to year_index.
        """
        from finance_engine import calc_exit_proceeds

        calendar_year = self.purchase_year + year_index
        tax = self._tax_engine
        loan_state = self._schedule.yearly_states[year_index - 1]

        # Estimate sale price (caller can override; here we use purchase price
        # as a neutral baseline — no appreciation assumption)
        # In the simulator, sale_price will be passed explicitly.
        sale_price = self.purchase_price  # placeholder; simulator overrides

        spec_tax = tax.calc_speculation_tax(
            sale_price=sale_price,
            original_purchase_price=self.purchase_price,
            cumulative_afa_claimed=self._cumulative_afa,
            holding_years=year_index,
            annual_income_in_exit_year=self.personal_income,
            filing_status=self.filing_status,
            simulation_year=calendar_year,
        )

        exit_proceeds = calc_exit_proceeds(
            sale_price=sale_price,
            remaining_loan_balance=loan_state.closing_balance,
            speculation_tax=spec_tax["speculation_tax"],
        )

        return ExitResult(
            year_index=year_index,
            calendar_year=calendar_year,
            sale_price=sale_price,
            remaining_loan_balance=loan_state.closing_balance,
            cumulative_afa_at_exit=self._cumulative_afa,
            speculation_tax_result=spec_tax,
            exit_proceeds=exit_proceeds,
        )


# ---------------------------------------------------------------------------
# Output data structures
# ---------------------------------------------------------------------------

@dataclass
class YearlySnapshot:
    """
    Complete financial picture for one year of the simulation.
    All sub-dicts are the raw outputs from tax_engine / finance_engine
    so every number is traceable to its source function.
    """
    year_index: int
    calendar_year: int

    # Loan
    loan_state: object              # YearlyLoanState

    # Rental income
    actual_rent_annual: float
    rent_rule: dict
    deduction_ratio: float

    # Renovation
    renovation_this_year: float
    renovation_deductible: float
    renovation_capitalised: float
    rule_15pct: dict
    cumulative_renovation: float

    # AfA
    afa_result: dict
    cumulative_afa: float

    # Tax
    werbungskosten: dict
    verpachtung_income: float
    total_taxable_income: float
    tax_with_property: dict
    tax_without_property: dict
    tax_delta: float                # Positive = tax saving, Negative = extra tax

    # Cash flow
    cashflow: dict

    def to_dict(self) -> dict:
        """Flatten to a dict for output_formatter consumption."""
        ls = self.loan_state
        return {
            "year_index": self.year_index,
            "calendar_year": self.calendar_year,
            # Loan
            "interest_paid": ls.annual_interest,
            "principal_paid": ls.annual_principal,
            "sondertilgung": ls.sondertilgung,
            "loan_balance_end": ls.closing_balance,
            "is_refi_year": ls.is_refi_year,
            "interest_rate": ls.interest_rate,
            # Rent
            "actual_rent": self.actual_rent_annual,
            "rent_zone": self.rent_rule["zone"],
            "deduction_ratio": self.deduction_ratio,
            # Renovation
            "renovation_spent": self.renovation_this_year,
            "renovation_deductible": self.renovation_deductible,
            "renovation_capitalised": self.renovation_capitalised,
            "flag_15pct_triggered": self.rule_15pct["triggered"],
            "cumulative_renovation": self.cumulative_renovation,
            # AfA
            "afa_building": self.afa_result["building_afa"],
            "afa_movable": self.afa_result["movable_afa"],
            "afa_total": self.afa_result["total_afa"],
            "cumulative_afa": self.cumulative_afa,
            # Tax
            "werbungskosten_total": self.werbungskosten["total_deductible"],
            "verpachtung_income": self.verpachtung_income,
            "total_taxable_income": self.total_taxable_income,
            "tax_with_property": self.tax_with_property["total_tax"],
            "tax_without_property": self.tax_without_property["total_tax"],
            "tax_delta": self.tax_delta,
            # Cash flow
            "net_cashflow": self.cashflow["net_cashflow"],
            "operating_cashflow": self.cashflow["operating_cashflow"],
            "financing_cashflow": self.cashflow["financing_cashflow"],
            "flag_negative_cashflow": self.cashflow["flag_negative_cashflow"],
        }


@dataclass
class ExitResult:
    """Results from selling the property at a specific year."""
    year_index: int
    calendar_year: int
    sale_price: float
    remaining_loan_balance: float
    cumulative_afa_at_exit: float
    speculation_tax_result: dict
    exit_proceeds: dict

    def to_dict(self) -> dict:
        return {
            "exit_year_index": self.year_index,
            "exit_calendar_year": self.calendar_year,
            "sale_price": self.sale_price,
            "remaining_loan_balance": self.remaining_loan_balance,
            "cumulative_afa_at_exit": self.cumulative_afa_at_exit,
            "speculation_tax": self.speculation_tax_result["speculation_tax"],
            "tax_free_exit": self.speculation_tax_result["tax_free"],
            "net_exit_proceeds": self.exit_proceeds["net_proceeds"],
            "gross_exit_proceeds": self.exit_proceeds["gross_proceeds"],
            "selling_costs": self.exit_proceeds["selling_costs"],
        }
