"""
action_space.py
---------------
Defines all actions available to the RL Agent.

Design:
  - Actions are discretized into a flat integer index space
    (required by PPO/DQN which expect Discrete(N) action spaces)
  - Each integer maps to an (ActionType, params_dict) pair
  - action_mask.py uses the same mapping to produce the boolean mask
  - action_engine.py uses the same mapping to execute the chosen action

Flat action index layout (Phase 1):
  0               → DO_NOTHING
  1..12           → BUY_PROPERTY  (3 LTV × 4 rate = 12 variants)
  13..16          → START_RENTING (4 rental_ratio variants)
  17..20          → ADJUST_RENT   (4 rental_ratio variants)
  21..24          → DO_RENOVATION (4 amount variants)
  25..26          → REFINANCE     (2 new_rate variants)
  27..28          → EXTRA_REPAYMENT (2 amount variants)
  29..33          → SELL_PROPERTY (5 sale_price_multiplier variants)

Total: 34 discrete actions (Phase 1)
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import IntEnum
from typing import Any


# ---------------------------------------------------------------------------
# Action types
# ---------------------------------------------------------------------------

class ActionType(IntEnum):
    DO_NOTHING      = 0
    BUY_PROPERTY    = 1
    START_RENTING   = 2
    ADJUST_RENT     = 3
    DO_RENOVATION   = 4
    REFINANCE       = 5
    EXTRA_REPAYMENT = 6
    SELL_PROPERTY   = 7


# ---------------------------------------------------------------------------
# Parameter grids (matching V1.0 spec discrete levels)
# ---------------------------------------------------------------------------

# BUY_PROPERTY parameters
BUY_LTV_LEVELS            = [0.70, 0.80, 0.90]
BUY_RATE_LEVELS           = [0.025, 0.035, 0.040, 0.050]
BUY_SONDERTILGUNG_DEFAULT = 0.01          # fixed at 1% for Phase 1 simplicity
BUY_LAND_RATIO_DEFAULT    = 0.25          # fixed default
BUY_ASSET_SPLIT_DEFAULT   = 10_000.0     # fixed default (kitchen etc.)
BUY_NEBENKOSTEN_RATE      = 0.12         # conservative estimate

# BUY: assumed purchase_price injected at execution time from user config
# The action only encodes the *financing structure*, not the price
BUY_VARIANTS = [
    {"ltv": ltv, "annual_rate": rate,
     "sondertilgung_rate": BUY_SONDERTILGUNG_DEFAULT,
     "land_ratio": BUY_LAND_RATIO_DEFAULT,
     "asset_split": BUY_ASSET_SPLIT_DEFAULT}
    for ltv in BUY_LTV_LEVELS
    for rate in BUY_RATE_LEVELS
]  # 3 × 4 = 12 variants

# START_RENTING / ADJUST_RENT parameters
RENT_RATIO_LEVELS = [0.66, 0.80, 0.95, 1.00]   # fraction of market rent
RENT_VARIANTS     = [{"rental_ratio": r} for r in RENT_RATIO_LEVELS]

# DO_RENOVATION parameters (EUR amounts)
# Relative to a €400k property: 10k/20k = safe, 40k borderline, 65k triggers 15% rule
RENOVATION_AMOUNT_LEVELS = [10_000, 20_000, 40_000, 65_000]
RENOVATION_VARIANTS      = [{"amount": a} for a in RENOVATION_AMOUNT_LEVELS]

# REFINANCE parameters (new annual rate)
REFI_RATE_LEVELS = [0.025, 0.030]
REFI_VARIANTS    = [{"new_rate": r} for r in REFI_RATE_LEVELS]

# EXTRA_REPAYMENT parameters (fraction of original principal)
SONDER_RATE_LEVELS = [0.02, 0.05]
SONDER_VARIANTS    = [{"sondertilgung_rate": r} for r in SONDER_RATE_LEVELS]

# SELL_PROPERTY parameters (sale price as multiple of purchase price)
SALE_MULTIPLIER_LEVELS = [0.90, 1.00, 1.10, 1.20, 1.30]
SELL_VARIANTS          = [{"sale_price_multiplier": m}
                          for m in SALE_MULTIPLIER_LEVELS]


# ---------------------------------------------------------------------------
# Action dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Action:
    action_type: ActionType
    params: dict[str, Any]
    action_index: int   # flat integer index used by gym Discrete space

    def __repr__(self) -> str:
        p_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"Action({self.action_type.name}[{self.action_index}] {p_str})"

    def __hash__(self):
        return hash(self.action_index)

    def __eq__(self, other):
        if isinstance(other, Action):
            return self.action_index == other.action_index
        return NotImplemented


# ---------------------------------------------------------------------------
# Action registry: flat index → Action object
# ---------------------------------------------------------------------------

def _build_action_list() -> list[Action]:
    actions = []
    idx = 0

    def add(atype: ActionType, params: dict) -> None:
        nonlocal idx
        actions.append(Action(action_type=atype, params=params, action_index=idx))
        idx += 1

    # 0: DO_NOTHING
    add(ActionType.DO_NOTHING, {})

    # 1–12: BUY_PROPERTY
    for p in BUY_VARIANTS:
        add(ActionType.BUY_PROPERTY, p)

    # 13–16: START_RENTING
    for p in RENT_VARIANTS:
        add(ActionType.START_RENTING, p)

    # 17–20: ADJUST_RENT
    for p in RENT_VARIANTS:
        add(ActionType.ADJUST_RENT, p)

    # 21–24: DO_RENOVATION
    for p in RENOVATION_VARIANTS:
        add(ActionType.DO_RENOVATION, p)

    # 25–26: REFINANCE
    for p in REFI_VARIANTS:
        add(ActionType.REFINANCE, p)

    # 27–28: EXTRA_REPAYMENT
    for p in SONDER_VARIANTS:
        add(ActionType.EXTRA_REPAYMENT, p)

    # 29–33: SELL_PROPERTY
    for p in SELL_VARIANTS:
        add(ActionType.SELL_PROPERTY, p)

    return actions


ALL_ACTIONS: list[Action] = _build_action_list()
N_ACTIONS: int = len(ALL_ACTIONS)

# Index lookup
_INDEX_TO_ACTION: dict[int, Action] = {a.action_index: a for a in ALL_ACTIONS}

# Type lookup: ActionType → list of Actions of that type
_TYPE_TO_ACTIONS: dict[ActionType, list[Action]] = {}
for _a in ALL_ACTIONS:
    _TYPE_TO_ACTIONS.setdefault(_a.action_type, []).append(_a)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_action(index: int) -> Action:
    """Retrieve an Action by its flat integer index."""
    if index not in _INDEX_TO_ACTION:
        raise ValueError(f"Action index {index} out of range [0, {N_ACTIONS-1}]")
    return _INDEX_TO_ACTION[index]


def actions_of_type(action_type: ActionType) -> list[Action]:
    """All actions of a given type."""
    return _TYPE_TO_ACTIONS.get(action_type, [])


def indices_of_type(action_type: ActionType) -> list[int]:
    """Flat indices of all actions of a given type."""
    return [a.action_index for a in actions_of_type(action_type)]


def action_summary() -> str:
    """Human-readable summary of the action space."""
    lines = [f"Total actions: {N_ACTIONS}", ""]
    for atype in ActionType:
        acts = actions_of_type(atype)
        lines.append(f"  {atype.name:20s}: {len(acts):3d} variant(s)")
        for a in acts:
            p_str = ", ".join(f"{k}={v}" for k, v in a.params.items())
            lines.append(f"    [{a.action_index:2d}] {p_str or '—'}")
    return "\n".join(lines)
