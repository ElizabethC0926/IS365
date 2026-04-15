"""
action_mask.py
--------------
Computes a boolean mask over all N_ACTIONS for a given PersonalState.

The mask is True for legal actions, False for illegal ones.
PPO (MaskablePPO from sb3-contrib) uses this to set illegal action
logits to -inf before softmax, so the agent never selects them.

Rules encoded here:
  DO_NOTHING        : always legal
  BUY_PROPERTY      : has sufficient liquid cash for at least one variant
                      AND not at property limit
  START_RENTING     : has a vacant property
  ADJUST_RENT       : has a renting property
  DO_RENOVATION     : has any owned property
                      AND sufficient cash for each renovation amount
  REFINANCE         : has at least one loan with balance > 0
  EXTRA_REPAYMENT   : same as REFINANCE + enough cash for the payment
  SELL_PROPERTY     : has an owned property

Design:
  - Conservative: for BUY, we check if cash covers the LOWEST-LTV variant
    (highest equity requirement). If not, all BUY variants are masked.
    Exact feasibility per variant is checked again in action_engine.py.
  - Per-variant masking: within BUY_PROPERTY, each LTV×rate variant is
    individually masked based on whether cash covers that specific variant.
  - Immutable: returns a fresh np.ndarray, never modifies state.
"""

from __future__ import annotations
import numpy as np

from personal_state import PersonalState, PropertyState
from action_space import (
    ActionType, ALL_ACTIONS, N_ACTIONS,
    actions_of_type, BUY_NEBENKOSTEN_RATE,
)


def compute_mask(state: PersonalState) -> np.ndarray:
    """
    Return a boolean array of shape [N_ACTIONS].
    True  = action is legal in this state.
    False = action is illegal (will be masked by MaskablePPO).
    """
    mask = np.zeros(N_ACTIONS, dtype=bool)

    owned    = state.owned_properties()
    vacant   = state.vacant_properties()
    renting  = state.renting_properties()
    has_loan = any(p.current_loan_balance > 0 for p in owned)

    # ── DO_NOTHING ──────────────────────────────────────────────────────
    mask[0] = True

    # ── BUY_PROPERTY ────────────────────────────────────────────────────
    if not state.at_property_limit():
        for action in actions_of_type(ActionType.BUY_PROPERTY):
            ltv = action.params["ltv"]
            # We need a purchase_price to check affordability.
            # During training we use a canonical price from config.
            # At inference the user provides the actual price.
            # Use a conservative default here; action_engine does exact check.
            canonical_price = _get_canonical_purchase_price(state)
            equity_needed   = canonical_price * (1.0 - ltv)
            nebenkosten     = canonical_price * BUY_NEBENKOSTEN_RATE
            if state.liquid_cash >= equity_needed + nebenkosten:
                mask[action.action_index] = True

    # ── START_RENTING ────────────────────────────────────────────────────
    if vacant:
        for action in actions_of_type(ActionType.START_RENTING):
            mask[action.action_index] = True

    # ── ADJUST_RENT ──────────────────────────────────────────────────────
    if renting:
        for action in actions_of_type(ActionType.ADJUST_RENT):
            mask[action.action_index] = True

    # ── DO_RENOVATION ────────────────────────────────────────────────────
    if owned:
        for action in actions_of_type(ActionType.DO_RENOVATION):
            amount = action.params["amount"]
            if state.liquid_cash >= amount:
                mask[action.action_index] = True

    # ── REFINANCE ────────────────────────────────────────────────────────
    if has_loan:
        for action in actions_of_type(ActionType.REFINANCE):
            mask[action.action_index] = True

    # ── EXTRA_REPAYMENT ──────────────────────────────────────────────────
    if has_loan:
        for action in actions_of_type(ActionType.EXTRA_REPAYMENT):
            # sondertilgung_rate × original principal
            # We approximate original principal as current balance / 0.7
            # (conservative — exact check in action_engine)
            approx_principal = max(p.current_loan_balance for p in owned) / 0.70
            approx_payment   = approx_principal * action.params["sondertilgung_rate"]
            if state.liquid_cash >= approx_payment:
                mask[action.action_index] = True

    # ── SELL_PROPERTY ────────────────────────────────────────────────────
    if owned:
        for action in actions_of_type(ActionType.SELL_PROPERTY):
            mask[action.action_index] = True

    return mask


def legal_action_indices(state: PersonalState) -> list[int]:
    """Convenience: return list of legal action indices."""
    mask = compute_mask(state)
    return list(np.where(mask)[0])


def legal_action_count(state: PersonalState) -> int:
    return int(compute_mask(state).sum())


def mask_summary(state: PersonalState) -> str:
    """Human-readable breakdown of legal/illegal actions."""
    mask = compute_mask(state)
    lines = [f"Legal actions: {mask.sum()}/{N_ACTIONS}"]
    for atype in ActionType:
        from action_space import actions_of_type as _ato
        acts = _ato(atype)
        legal = [a for a in acts if mask[a.action_index]]
        lines.append(
            f"  {atype.name:20s}: "
            f"{len(legal)}/{len(acts)} legal"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

# Canonical purchase prices used during training when no specific price given.
# Covers a range of realistic German property prices.
_CANONICAL_PRICES = [200_000, 300_000, 400_000, 500_000, 600_000]

def _get_canonical_purchase_price(state: PersonalState) -> float:
    """
    Pick a representative purchase price for affordability checks
    during training.

    Logic: use 3× annual_income as a rough proxy for what the investor
    might target, clamped to the canonical range.
    """
    target = state.annual_income * 3.5
    # Clamp to canonical range
    clamped = max(_CANONICAL_PRICES[0],
                  min(_CANONICAL_PRICES[-1], target))
    # Round to nearest canonical price
    return min(_CANONICAL_PRICES, key=lambda p: abs(p - clamped))
