"""
world_model.py
--------------
The central simulation dispatcher. This is the "physics engine" of the MDP.

Each call to step() represents one calendar year passing:
  1. Execute the chosen action (action_engine)
  2. Advance the loan by one year (finance_engine)
  3. Compute annual tax (tax_engine)
  4. Detect FLAGS (flag_system logic, inline)
  5. Compute annual cash flow
  6. Update all cumulative fields on PropertyState
  7. Return (new_state, reward_components, info)

reward_components is a dict of labelled numbers — reward.py combines
them into a scalar. Keeping them separate makes Reward Shaping experiments
easy (just change the weights in reward.py without touching world_model).

Design:
  - Pure function: never mutates inputs.
  - All monetary results rounded to 2 decimal places.
  - Stores the full year snapshot in info["yearly_snapshot"] for
    Decision Log consumption.
"""

from __future__ import annotations
import copy

from personal_state import PersonalState, PropertyState
from action_space import Action, ActionType
from action_engine import execute as action_execute, ActionError
from tax_engine import TaxEngine
from finance_engine import calc_annual_cashflow


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def step(
    state: PersonalState,
    action: Action,
    tax_engine: TaxEngine,
    purchase_price: float | None = None,
    target_property_index: int = 0,
) -> tuple[PersonalState, dict, dict]:
    """
    Advance the simulation by one year.

    Parameters
    ----------
    state : PersonalState
        State at the START of this year (will not be mutated).
    action : Action
        The action chosen by the Agent for this year.
    tax_engine : TaxEngine
        Shared TaxEngine instance.
    purchase_price : float, optional
        Required only if action.action_type == BUY_PROPERTY.
    target_property_index : int
        Which property slot the action targets (Phase 1: always 0).

    Returns
    -------
    new_state : PersonalState
        State at the END of this year.
    reward_components : dict
        Labelled reward contributions (combined by reward.py).
    info : dict
        Full year snapshot for Decision Log and debugging.
    """
    # 1. Execute action
    try:
        mid_state, action_financials = action_execute(
            state, action, tax_engine,
            target_property_index, purchase_price
        )
    except ActionError as exc:
        # Illegal action despite mask — treat as DO_NOTHING with a penalty
        mid_state        = state.copy()
        action_financials = {
            "action": "ILLEGAL_FALLBACK",
            "cash_delta": 0.0,
            "description": f"Illegal action blocked: {exc}",
        }

    new_state = mid_state.copy()
    new_state.current_year  += 1
    new_state.years_elapsed += 1

    # 2–6. Process each owned property
    all_prop_snapshots = []
    total_annual_interest    = 0.0
    total_annual_cf          = 0.0
    total_tax_delta          = 0.0
    total_afa                = 0.0
    total_wk                 = 0.0
    any_flag_15pct           = False
    any_flag_rent_low        = False
    any_flag_tax_waste       = False
    any_flag_neg_cf          = False

    for i, prop in enumerate(new_state.properties):
        if not prop.is_owned():
            all_prop_snapshots.append(None)
            continue

        snap, updated_prop = _process_property_year(
            prop, new_state, tax_engine
        )
        new_state.properties[i] = updated_prop
        all_prop_snapshots.append(snap)

        # Accumulate totals
        total_annual_interest += snap["interest_paid"]
        total_annual_cf       += snap["net_cashflow"]
        total_tax_delta       += snap["tax_delta"]
        total_afa             += snap["afa_total"]
        total_wk              += snap["werbungskosten_total"]

        # FLAGS
        if snap["flag_15pct_triggered"]:  any_flag_15pct     = True
        if snap["flag_rent_too_low"]:     any_flag_rent_low  = True
        if snap["flag_neg_cf"]:           any_flag_neg_cf    = True
        if snap["flag_tax_waste"]:        any_flag_tax_waste = True

    # Add any action-level cash flows (buy/sell/renovation cash in/out)
    action_cash = action_financials.get("cash_delta", 0.0)

    # Reward components (raw numbers, not yet weighted)
    reward_components = {
        "annual_net_cashflow":    total_annual_cf,
        "action_cash_delta":      action_cash,
        "tax_delta":              total_tax_delta,
        "flag_15pct_hit":         float(any_flag_15pct),
        "flag_rent_too_low":      float(any_flag_rent_low),
        "flag_tax_waste":         float(any_flag_tax_waste),
        "flag_negative_cashflow": float(any_flag_neg_cf),
        # Exit reward is only non-zero when SELL action was taken
        "exit_net_proceeds": (
            action_financials.get("net_proceeds", 0.0)
            if action.action_type == ActionType.SELL_PROPERTY
            else 0.0
        ),
    }

    info = {
        "year":               new_state.current_year - 1,   # year just completed
        "action":             action_financials,
        "property_snapshots": all_prop_snapshots,
        "totals": {
            "net_cashflow":    total_annual_cf,
            "interest_paid":   total_annual_interest,
            "tax_delta":       total_tax_delta,
            "afa":             total_afa,
            "werbungskosten":  total_wk,
        },
        "flags": {
            "FLAG_15_PERCENT_HIT":   any_flag_15pct,
            "FLAG_RENT_TOO_LOW":     any_flag_rent_low,
            "FLAG_TAX_WASTE":        any_flag_tax_waste,
            "FLAG_NEGATIVE_CASHFLOW":any_flag_neg_cf,
        },
        "state_summary": new_state.summary(),
    }

    return new_state, reward_components, info


# ---------------------------------------------------------------------------
# Per-property annual processing
# ---------------------------------------------------------------------------

def _process_property_year(
    prop: PropertyState,
    state: PersonalState,
    tax_engine: TaxEngine,
) -> tuple[dict, PropertyState]:
    """
    Run one year of tax + finance calculations for a single property.
    Returns (snapshot_dict, updated_prop).
    """
    sim_year     = state.current_year
    years_since  = sim_year - prop.purchase_year - 1  # years completed at start

    # ── Loan: advance one year ──────────────────────────────────────────
    schedule = getattr(prop, "_loan_schedule", None)
    if schedule and years_since < len(schedule.yearly_states):
        loan_state = schedule.yearly_states[years_since]
        interest      = loan_state.annual_interest
        principal_pmt = loan_state.annual_principal
        sonder        = loan_state.sondertilgung
        new_balance   = loan_state.closing_balance
    else:
        # Loan fully repaid or no schedule
        interest = principal_pmt = sonder = 0.0
        new_balance = 0.0

    loan_payment = interest + principal_pmt   # regular annuity (excl. Sonder)

    # ── AfA ─────────────────────────────────────────────────────────────
    land_ratio  = getattr(prop, "_land_ratio",  0.25)
    asset_split = getattr(prop, "_asset_split", 0.0)
    building_val = prop.purchase_price * (1 - land_ratio) - asset_split

    afa_result = tax_engine.calc_afa(
        building_value   = max(0.0, building_val),
        movable_value    = asset_split,
        year_of_purchase = prop.purchase_year,
        simulation_year  = sim_year,
        building_type    = prop.building_type,
    )
    afa_total = afa_result["total_afa"]

    # ── 15% rule ─────────────────────────────────────────────────────────
    rule_15 = tax_engine.check_15pct_rule(
        renovation_cumulative_net = prop.cumulative_renovation,
        purchase_price_net        = prop.purchase_price,
        years_since_purchase      = years_since + 1,
        simulation_year           = sim_year,
    )
    reno_deductible = 0.0   # renovation cash already deducted in action_engine
    # If rule triggered, the renovation amount gets capitalised — increases
    # AfA base in subsequent years (simplified: we don't retroactively adjust)

    # ── Rent rule ────────────────────────────────────────────────────────
    rent_result = tax_engine.check_rent_rule(
        actual_rent_annual  = prop.current_rent_annual,
        market_rent_annual  = prop.market_rent_annual,
        simulation_year     = sim_year,
    )
    deduction_ratio = rent_result["deduction_ratio"]

    # ── Werbungskosten ──────────────────────────────────────────────────
    mgmt_costs      = prop.current_rent_annual * 0.02   # 2% of rent
    insurance_costs = prop.purchase_price * 0.001       # 0.1% of value
    wk_result = tax_engine.calc_werbungskosten(
        interest_paid        = interest,
        afa_total            = afa_total,
        renovation_deductible= reno_deductible,
        management_costs     = mgmt_costs,
        insurance_costs      = insurance_costs,
        other_costs          = 500.0,
        deduction_ratio      = deduction_ratio,
        simulation_year      = sim_year,
    )
    wk_total = wk_result["total_deductible"]

    # ── VuV income & income tax ─────────────────────────────────────────
    vuv_income = round(prop.current_rent_annual - wk_total, 2)
    total_zve  = max(0.0, state.annual_income + vuv_income)

    tax_with    = tax_engine.calc_income_tax(total_zve, sim_year,
                                              state.filing_status)
    tax_without = tax_engine.calc_income_tax(state.annual_income, sim_year,
                                              state.filing_status)
    tax_delta   = round(tax_without["total_tax"] - tax_with["total_tax"], 2)

    # ── Annual cash flow ─────────────────────────────────────────────────
    cf = calc_annual_cashflow(
        rental_income_gross  = prop.current_rent_annual,
        non_deductible_costs = 500.0,
        loan_payment         = loan_payment,
        tax_refund           = tax_delta,
        sondertilgung        = sonder,
    )
    state.liquid_cash = round(state.liquid_cash + cf["net_cashflow"], 2)

    # ── FLAG: tax waste ──────────────────────────────────────────────────
    flag_tax_waste = (
        vuv_income < 0 and abs(vuv_income) > state.annual_income
    )

    # ── Update PropertyState ─────────────────────────────────────────────
    new_prop = copy.copy(prop)
    new_prop.current_loan_balance = new_balance
    new_prop.cumulative_afa       = round(prop.cumulative_afa + afa_total, 2)
    new_prop.years_owned          = prop.years_owned + 1

    # Preserve schedule and other side-channel attrs
    for attr in ("_loan_schedule", "_land_ratio", "_asset_split",
                 "_sondertilgung_rate"):
        val = getattr(prop, attr, None)
        if val is not None:
            object.__setattr__(new_prop, attr, val)

    snapshot = {
        # Loan
        "interest_paid":    interest,
        "principal_paid":   principal_pmt,
        "sondertilgung":    sonder,
        "loan_balance_end": new_balance,
        # Rent
        "rent_annual":      prop.current_rent_annual,
        "rent_zone":        rent_result["zone"],
        "deduction_ratio":  deduction_ratio,
        # Tax
        "afa_total":             afa_total,
        "werbungskosten_total":  wk_total,
        "vuv_income":            vuv_income,
        "tax_delta":             tax_delta,
        # Cash flow
        "net_cashflow":  cf["net_cashflow"],
        # Flags
        "flag_15pct_triggered": rule_15["triggered"],
        "flag_rent_too_low":    rent_result["flag_rent_too_low"],
        "flag_neg_cf":          cf["flag_negative_cashflow"],
        "flag_tax_waste":       flag_tax_waste,
        # Meta
        "law_refs": {
            "15pct": rule_15["law_ref"],
            "rent":  "§21 Abs.2 EStG",
            "afa":   afa_result["law_ref"],
        },
    }

    return snapshot, new_prop
