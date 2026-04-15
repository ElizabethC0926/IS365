"""
action_engine.py
----------------
Executes individual actions and returns an updated PersonalState.

Each execute_* function:
  1. Validates preconditions (raises ActionError if violated)
  2. Updates the relevant PropertyState and PersonalState fields
  3. Returns (new_state, financials_dict)

financials_dict contains the immediate cash flows from the action
(not the full year's tax calculation — that happens in world_model.py).

Design:
  - Never mutates the input state. Always works on state.copy().
  - Tax calculations (speculation tax, Grunderwerbsteuer) are delegated
    to tax_engine.py where needed.
  - finance_engine.py provides the loan schedule on BUY.
  - All monetary values in EUR.
"""

from __future__ import annotations
from dataclasses import replace
import copy

from personal_state import PersonalState, PropertyState
from action_space import Action, ActionType
from tax_engine import TaxEngine
from finance_engine import (
    calc_purchase_costs,
    calc_equity_and_loan,
    build_amortization_schedule,
)


class ActionError(Exception):
    """Raised when an action's preconditions are not met."""


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def execute(
    state: PersonalState,
    action: Action,
    tax_engine: TaxEngine,
    target_property_index: int = 0,
    purchase_price: float | None = None,
) -> tuple[PersonalState, dict]:
    """
    Execute an action and return (new_state, financials).

    Parameters
    ----------
    state : PersonalState
        Current state (will not be mutated).
    action : Action
        The chosen action (from action_space.ALL_ACTIONS).
    tax_engine : TaxEngine
        Shared tax engine instance.
    target_property_index : int
        Which property slot to target for multi-property actions
        (Phase 1: always 0).
    purchase_price : float, optional
        Required for BUY_PROPERTY. If None, raises ActionError.

    Returns
    -------
    new_state : PersonalState
        Updated state after the action.
    financials : dict
        Immediate financial flows from this action (for reward calculation).
    """
    new_state = state.copy()

    dispatch = {
        ActionType.DO_NOTHING:      _execute_do_nothing,
        ActionType.BUY_PROPERTY:    _execute_buy,
        ActionType.START_RENTING:   _execute_start_renting,
        ActionType.ADJUST_RENT:     _execute_adjust_rent,
        ActionType.DO_RENOVATION:   _execute_renovation,
        ActionType.REFINANCE:       _execute_refinance,
        ActionType.EXTRA_REPAYMENT: _execute_extra_repayment,
        ActionType.SELL_PROPERTY:   _execute_sell,
    }

    fn = dispatch.get(action.action_type)
    if fn is None:
        raise ActionError(f"Unknown action type: {action.action_type}")

    return fn(
        new_state, action, tax_engine,
        target_property_index, purchase_price
    )


# ---------------------------------------------------------------------------
# DO_NOTHING
# ---------------------------------------------------------------------------

def _execute_do_nothing(
    state, action, tax_engine, prop_idx, purchase_price
) -> tuple[PersonalState, dict]:
    return state, {
        "action": "DO_NOTHING",
        "cash_delta": 0.0,
        "description": "No action taken this year.",
    }


# ---------------------------------------------------------------------------
# BUY_PROPERTY
# ---------------------------------------------------------------------------

def _execute_buy(
    state, action, tax_engine, prop_idx, purchase_price
) -> tuple[PersonalState, dict]:
    if purchase_price is None or purchase_price <= 0:
        raise ActionError("purchase_price must be provided for BUY_PROPERTY")
    if state.at_property_limit():
        raise ActionError("Property limit reached")

    p = action.params
    ltv              = p["ltv"]
    annual_rate      = p["annual_rate"]
    sondertilgung    = p["sondertilgung_rate"]
    land_ratio       = p["land_ratio"]
    asset_split      = p["asset_split"]

    # --- Purchase costs (Grunderwerbsteuer etc.) ---
    # Use a default state for GrESt calculation;
    # env.py passes german_state via config
    gst_result = tax_engine.calc_grunderwerbsteuer(
        purchase_price,
        state.properties[prop_idx].german_state
        if state.properties[prop_idx].status == "none"
        else "Bayern",    # fallback; overridden by env config
        state.current_year,
    )
    pc = calc_purchase_costs(
        purchase_price,
        grunderwerbsteuer_rate=gst_result["rate"],
    )
    eq = calc_equity_and_loan(
        purchase_price,
        pc["total_nebenkosten"],
        equity_amount=purchase_price * (1 - ltv),
    )

    equity_out   = purchase_price * (1 - ltv)
    nebenkosten  = pc["total_nebenkosten"]
    total_cash_out = equity_out + nebenkosten

    if state.liquid_cash < total_cash_out:
        raise ActionError(
            f"Insufficient cash: need €{total_cash_out:,.0f}, "
            f"have €{state.liquid_cash:,.0f}"
        )

    loan_amount = eq["loan_amount"]

    # Build amortization schedule (stored on PropertyState for world_model)
    # We store the schedule as a simple dict; world_model uses it each year
    schedule = build_amortization_schedule(
        principal=loan_amount,
        annual_rate=annual_rate,
        holding_years=15,        # max horizon; world_model truncates at exit
        sondertilgung_rate=sondertilgung,
        purchase_year=state.current_year,
    )

    # Market rent estimate: ~4.5% of purchase price per year
    market_rent = purchase_price * 0.045

    # Create new PropertyState
    new_prop = PropertyState(
        status               = "owned_vacant",
        german_state         = state.properties[prop_idx].german_state,
        building_type        = p.get("building_type", "standard"),
        purchase_year        = state.current_year,
        purchase_price       = purchase_price,
        market_rent_annual   = market_rent,
        current_loan_balance = loan_amount,
        annual_rate          = annual_rate,
        cumulative_afa       = 0.0,
        cumulative_renovation= 0.0,
        current_rent_annual  = 0.0,
        years_owned          = 0,
    )

    # Attach schedule as a side-channel attribute
    # (dataclass doesn't support extra fields, so we use object.__setattr__)
    object.__setattr__(new_prop, "_loan_schedule", schedule)
    object.__setattr__(new_prop, "_land_ratio",    land_ratio)
    object.__setattr__(new_prop, "_asset_split",   asset_split)
    object.__setattr__(new_prop, "_sondertilgung_rate", sondertilgung)

    # Update state
    state.liquid_cash -= total_cash_out
    state.properties[prop_idx] = new_prop

    return state, {
        "action":        "BUY_PROPERTY",
        "purchase_price": purchase_price,
        "loan_amount":   loan_amount,
        "ltv":           ltv,
        "equity_out":    equity_out,
        "nebenkosten":   nebenkosten,
        "cash_delta":    -total_cash_out,
        "annual_rate":   annual_rate,
        "description":   (
            f"Bought €{purchase_price:,.0f} property. "
            f"Loan €{loan_amount:,.0f} @ {annual_rate:.1%}. "
            f"Cash out €{total_cash_out:,.0f}."
        ),
    }


# ---------------------------------------------------------------------------
# START_RENTING
# ---------------------------------------------------------------------------

def _execute_start_renting(
    state, action, tax_engine, prop_idx, purchase_price
) -> tuple[PersonalState, dict]:
    prop = _require_vacant(state, prop_idx)

    rental_ratio       = action.params["rental_ratio"]
    new_rent           = prop.market_rent_annual * rental_ratio
    new_prop           = copy.copy(prop)
    new_prop.status    = "owned_renting"
    new_prop.current_rent_annual = new_rent
    state.properties[prop_idx] = new_prop

    return state, {
        "action":       "START_RENTING",
        "rental_ratio": rental_ratio,
        "new_rent":     new_rent,
        "cash_delta":   0.0,
        "description":  (
            f"Started renting at €{new_rent:,.0f}/yr "
            f"({rental_ratio:.0%} of market)."
        ),
    }


# ---------------------------------------------------------------------------
# ADJUST_RENT
# ---------------------------------------------------------------------------

def _execute_adjust_rent(
    state, action, tax_engine, prop_idx, purchase_price
) -> tuple[PersonalState, dict]:
    prop = _require_renting(state, prop_idx)

    rental_ratio                 = action.params["rental_ratio"]
    new_rent                     = prop.market_rent_annual * rental_ratio
    new_prop                     = copy.copy(prop)
    new_prop.current_rent_annual = new_rent
    state.properties[prop_idx]   = new_prop

    return state, {
        "action":       "ADJUST_RENT",
        "rental_ratio": rental_ratio,
        "old_rent":     prop.current_rent_annual,
        "new_rent":     new_rent,
        "cash_delta":   0.0,
        "description":  (
            f"Adjusted rent to €{new_rent:,.0f}/yr "
            f"({rental_ratio:.0%} of market)."
        ),
    }


# ---------------------------------------------------------------------------
# DO_RENOVATION
# ---------------------------------------------------------------------------

def _execute_renovation(
    state, action, tax_engine, prop_idx, purchase_price
) -> tuple[PersonalState, dict]:
    prop   = _require_owned(state, prop_idx)
    amount = action.params["amount"]

    if state.liquid_cash < amount:
        raise ActionError(
            f"Insufficient cash for renovation: "
            f"need €{amount:,.0f}, have €{state.liquid_cash:,.0f}"
        )

    # Check 15% rule (for decision log — actual tax impact in world_model)
    years_since_purchase = state.current_year - prop.purchase_year
    rule_result = tax_engine.check_15pct_rule(
        renovation_cumulative_net = prop.cumulative_renovation + amount,
        purchase_price_net        = prop.purchase_price,
        years_since_purchase      = years_since_purchase,
        simulation_year           = state.current_year,
    )

    new_prop = copy.copy(prop)
    new_prop.cumulative_renovation += amount
    state.properties[prop_idx]     = new_prop
    state.liquid_cash             -= amount

    return state, {
        "action":             "DO_RENOVATION",
        "amount":             amount,
        "cash_delta":         -amount,
        "flag_15pct_hit":     rule_result["triggered"],
        "deductible_now":     not rule_result["triggered"],
        "law_ref":            rule_result["law_ref"],
        "description": (
            f"Renovation €{amount:,.0f}. "
            + ("⚠ 15% rule triggered — costs capitalised."
               if rule_result["triggered"]
               else "Safe: costs deductible as Werbungskosten.")
        ),
    }


# ---------------------------------------------------------------------------
# REFINANCE
# ---------------------------------------------------------------------------

def _execute_refinance(
    state, action, tax_engine, prop_idx, purchase_price
) -> tuple[PersonalState, dict]:
    prop = _require_owned(state, prop_idx)
    if prop.current_loan_balance <= 0:
        raise ActionError("No outstanding loan to refinance")

    new_rate    = action.params["new_rate"]
    old_rate    = prop.annual_rate
    new_prop    = copy.copy(prop)
    new_prop.annual_rate = new_rate

    # Rebuild amortization schedule from current balance
    remaining_years = max(1, 15 - (state.current_year - prop.purchase_year))
    sonder = getattr(prop, "_sondertilgung_rate", 0.0)
    new_schedule = build_amortization_schedule(
        principal       = prop.current_loan_balance,
        annual_rate     = new_rate,
        holding_years   = remaining_years,
        sondertilgung_rate = sonder,
        purchase_year   = state.current_year - 1,
    )
    object.__setattr__(new_prop, "_loan_schedule", new_schedule)

    state.properties[prop_idx] = new_prop

    return state, {
        "action":    "REFINANCE",
        "old_rate":  old_rate,
        "new_rate":  new_rate,
        "balance":   prop.current_loan_balance,
        "cash_delta": 0.0,
        "description": (
            f"Refinanced €{prop.current_loan_balance:,.0f} "
            f"from {old_rate:.1%} to {new_rate:.1%}."
        ),
    }


# ---------------------------------------------------------------------------
# EXTRA_REPAYMENT (Sondertilgung)
# ---------------------------------------------------------------------------

def _execute_extra_repayment(
    state, action, tax_engine, prop_idx, purchase_price
) -> tuple[PersonalState, dict]:
    prop = _require_owned(state, prop_idx)
    if prop.current_loan_balance <= 0:
        raise ActionError("No outstanding loan for extra repayment")

    # sondertilgung_rate × current balance (simplified: rate × balance)
    rate   = action.params["sondertilgung_rate"]
    amount = min(prop.current_loan_balance, prop.current_loan_balance * rate)
    amount = round(amount, 2)

    if state.liquid_cash < amount:
        raise ActionError(
            f"Insufficient cash for extra repayment: "
            f"need €{amount:,.0f}, have €{state.liquid_cash:,.0f}"
        )

    new_prop = copy.copy(prop)
    new_prop.current_loan_balance -= amount
    new_prop.current_loan_balance  = max(0.0, new_prop.current_loan_balance)
    state.properties[prop_idx]    = new_prop
    state.liquid_cash            -= amount

    return state, {
        "action":     "EXTRA_REPAYMENT",
        "amount":     amount,
        "rate":       rate,
        "remaining":  new_prop.current_loan_balance,
        "cash_delta": -amount,
        "description": (
            f"Extra repayment €{amount:,.0f}. "
            f"Remaining loan: €{new_prop.current_loan_balance:,.0f}."
        ),
    }


# ---------------------------------------------------------------------------
# SELL_PROPERTY
# ---------------------------------------------------------------------------

def _execute_sell(
    state, action, tax_engine, prop_idx, purchase_price
) -> tuple[PersonalState, dict]:
    prop = _require_owned(state, prop_idx)

    multiplier  = action.params["sale_price_multiplier"]
    sale_price  = round(prop.purchase_price * multiplier, 2)
    holding_yrs = state.current_year - prop.purchase_year

    # Speculation tax
    spec = tax_engine.calc_speculation_tax(
        sale_price                = sale_price,
        original_purchase_price   = prop.purchase_price,
        cumulative_afa_claimed    = prop.cumulative_afa,
        holding_years             = holding_yrs,
        annual_income_in_exit_year= state.annual_income,
        filing_status             = state.filing_status,
        simulation_year           = state.current_year,
    )

    # Selling costs (Makler + Notar)
    selling_costs = round(sale_price * (0.0357 + 0.010), 2)
    net_proceeds  = round(
        sale_price
        - prop.current_loan_balance
        - selling_costs
        - spec["speculation_tax"],
        2
    )

    # Mark property as sold
    new_prop         = copy.copy(prop)
    new_prop.status  = "sold"
    new_prop.current_loan_balance = 0.0
    new_prop.current_rent_annual  = 0.0
    state.properties[prop_idx]   = new_prop
    state.liquid_cash            += net_proceeds

    return state, {
        "action":           "SELL_PROPERTY",
        "sale_price":       sale_price,
        "multiplier":       multiplier,
        "holding_years":    holding_yrs,
        "selling_costs":    selling_costs,
        "loan_repaid":      prop.current_loan_balance,
        "speculation_tax":  spec["speculation_tax"],
        "tax_free":         spec["tax_free"],
        "net_proceeds":     net_proceeds,
        "cash_delta":       net_proceeds,
        "law_ref":          spec["law_ref"],
        "description": (
            f"Sold for €{sale_price:,.0f} "
            f"(×{multiplier} of purchase price). "
            f"Held {holding_yrs}yr. "
            + ("Tax-free exit." if spec["tax_free"]
               else f"Spec tax €{spec['speculation_tax']:,.0f}.")
            + f" Net proceeds €{net_proceeds:,.0f}."
        ),
    }


# ---------------------------------------------------------------------------
# Precondition helpers
# ---------------------------------------------------------------------------

def _require_owned(state: PersonalState, idx: int) -> PropertyState:
    if idx >= len(state.properties):
        raise ActionError(f"Property index {idx} out of range")
    prop = state.properties[idx]
    if not prop.is_owned():
        raise ActionError(
            f"Property {idx} is not owned (status={prop.status})"
        )
    return prop


def _require_vacant(state: PersonalState, idx: int) -> PropertyState:
    prop = _require_owned(state, idx)
    if prop.status != "owned_vacant":
        raise ActionError(
            f"Property {idx} is not vacant (status={prop.status})"
        )
    return prop


def _require_renting(state: PersonalState, idx: int) -> PropertyState:
    prop = _require_owned(state, idx)
    if prop.status != "owned_renting":
        raise ActionError(
            f"Property {idx} is not renting (status={prop.status})"
        )
    return prop
