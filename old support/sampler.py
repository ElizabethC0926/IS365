"""
sampler.py
----------
Generates all strategy parameter combinations from a user's fixed inputs.

The user fixes:
    purchase_price, state, purchase_year, personal_income,
    filing_status, market_rent_annual, building_type

The sampler varies all decision dimensions defined in V1.0:
    Core Strategic  : usage, holding_years, rental_ratio
    Tax Tactical    : renovation_year, renovation_amount, asset_split, land_ratio
    Financial       : equity_amount, annual_rate, sondertilgung_rate, refi_year/rate

Each combination = one row in the results table.

Sampling strategy:
  - Discrete grids per dimension (matching V1.0 spec levels).
  - Full combinatorial product by default.
  - Special forced groups (collision tests, exit timing pairs) are tagged
    with group_tag so analyzer.py can isolate them.
  - Invalid combinations filtered out by validators.py before returning.

Combinatorial size warning:
  Full grid can be large. Default settings produce ~2,000-5,000 combinations
  for a typical property. The RL layer (Layer 2) is what makes this tractable
  at scale; for MVP the full grid is fine.
"""

from __future__ import annotations
import itertools
from dataclasses import dataclass, field
from typing import Optional
from validators import validate


# ---------------------------------------------------------------------------
# Dimension grids (matching V1.0 spec)
# ---------------------------------------------------------------------------

# V1.0 Core Strategic
USAGE_LEVELS = ["full_rental", "partial"]          # own_use excluded (no investment case)
HOLDING_YEARS_LEVELS = [5, 7, 9, 10, 11, 12, 15]  # 9/11 split around §23 threshold
RENTAL_RATIO_LEVELS = [1.0, 0.90, 0.80, 0.66]     # 1.0=market, 0.66=threshold

# V1.0 Tax Tactical
# renovation_year: relative to purchase (1-based)
# Splits around the 3-year window for the 15% rule
RENOVATION_YEAR_LEVELS = [2, 4, 7]                # yr2=dangerous, yr4=safe, yr7=late
RENOVATION_AMOUNT_LEVELS = [0, 15_000, 40_000, 65_000]  # 0=none, 65k triggers 15% on 400k
ASSET_SPLIT_LEVELS = [0, 10_000, 20_000, 30_000]  # movables split
LAND_RATIO_LEVELS = [0.20, 0.30, 0.40]            # typical German range

# V1.0 Financial Leverage
# equity expressed as fraction of purchase_price for portability
EQUITY_RATIO_LEVELS = [0.10, 0.20, 0.30, 0.50]   # LTV ~90%, 80%, 70%, 50%
ANNUAL_RATE_LEVELS = [0.025, 0.035, 0.040, 0.050]
SONDERTILGUNG_LEVELS = [0.0, 0.01, 0.02, 0.05]
# Refi: None = no refi; (year, rate) pairs
REFI_OPTIONS = [
    (None, None),           # no refinancing
    (5, 0.025),             # refi at year 5 to lower rate
    (7, 0.030),
]

# Dimensions and their level lists (order matters for itertools.product)
_GRID_DIMENSIONS = [
    ("usage",               USAGE_LEVELS),
    ("holding_years",       HOLDING_YEARS_LEVELS),
    ("rental_ratio",        RENTAL_RATIO_LEVELS),
    ("renovation_year_rel", RENOVATION_YEAR_LEVELS),   # relative; clamped below
    ("renovation_amount",   RENOVATION_AMOUNT_LEVELS),
    ("asset_split",         ASSET_SPLIT_LEVELS),
    ("land_ratio",          LAND_RATIO_LEVELS),
    ("equity_ratio",        EQUITY_RATIO_LEVELS),      # converted to EUR below
    ("annual_rate",         ANNUAL_RATE_LEVELS),
    ("sondertilgung_rate",  SONDERTILGUNG_LEVELS),
    ("refi_option",         REFI_OPTIONS),
]


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class SamplerResult:
    total_generated: int
    total_valid: int
    total_invalid: int
    cases: list[dict]
    group_counts: dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"Generated : {self.total_generated:>6,}",
            f"Valid     : {self.total_valid:>6,}",
            f"Invalid   : {self.total_invalid:>6,}",
            "Groups    :",
        ]
        for group, count in sorted(self.group_counts.items()):
            lines.append(f"  {group:<35}: {count:>5,}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------

def _build_case(
    fixed: dict,
    combo: dict,
    group_tag: str = "random",
) -> Optional[dict]:
    """
    Merge fixed inputs with one combination of decision variables.
    Returns a validated params dict, or None if invalid.
    """
    purchase_price = fixed["purchase_price"]

    # Convert equity_ratio → equity_amount
    equity_amount = round(purchase_price * combo["equity_ratio"], 2)

    # Clamp renovation_year to holding_years
    reno_year = min(combo["renovation_year_rel"], combo["holding_years"])
    # If renovation_amount > 0 but year got clamped to last year, keep it
    # (simulator handles it; just ensure it's within range)

    # Unpack refi option
    refi_year, refi_rate = combo["refi_option"]
    # Refi year must be < holding_years
    if refi_year is not None and refi_year >= combo["holding_years"]:
        refi_year = None
        refi_rate = None

    params = {
        # Fixed
        **{k: fixed[k] for k in (
            "purchase_price", "state", "purchase_year",
            "personal_income", "filing_status",
            "market_rent_annual", "building_type",
        )},
        # Decision variables
        "usage":              combo["usage"],
        "holding_years":      combo["holding_years"],
        "rental_ratio":       combo["rental_ratio"],
        "renovation_year":    reno_year,
        "renovation_amount":  combo["renovation_amount"],
        "asset_split":        combo["asset_split"],
        "land_ratio":         combo["land_ratio"],
        "equity_amount":      equity_amount,
        "annual_rate":        combo["annual_rate"],
        "sondertilgung_rate": combo["sondertilgung_rate"],
        "refi_year":          refi_year,
        "refi_rate":          refi_rate,
        # Metadata
        "_group_tag":         group_tag,
    }

    result = validate(params)
    if not result.valid:
        return None

    return params


# ---------------------------------------------------------------------------
# Forced groups (for analysis / paper scenarios)
# ---------------------------------------------------------------------------

def _forced_15pct_collision(fixed: dict) -> list[dict]:
    """
    Force cases that trigger the 15% rule.
    renovation in year 1 or 2, amount > 15% of purchase price.
    """
    cases = []
    trigger_amount = fixed["purchase_price"] * 0.16   # just over the limit
    for holding in [10, 12, 15]:
        for rate in [0.030, 0.040]:
            combo = dict(
                usage="full_rental",
                holding_years=holding,
                rental_ratio=1.0,
                renovation_year_rel=2,
                renovation_amount=round(trigger_amount, 0),
                asset_split=10_000,
                land_ratio=0.25,
                equity_ratio=0.25,
                annual_rate=rate,
                sondertilgung_rate=0.0,
                refi_option=(None, None),
            )
            case = _build_case(fixed, combo, group_tag="collision_15pct_triggered")
            if case:
                cases.append(case)
    return cases


def _forced_15pct_avoided(fixed: dict) -> list[dict]:
    """
    Mirror of collision group but renovation pushed to year 4 (safe).
    """
    cases = []
    trigger_amount = fixed["purchase_price"] * 0.16
    for holding in [10, 12, 15]:
        for rate in [0.030, 0.040]:
            combo = dict(
                usage="full_rental",
                holding_years=holding,
                rental_ratio=1.0,
                renovation_year_rel=4,          # outside 3-year window
                renovation_amount=round(trigger_amount, 0),
                asset_split=10_000,
                land_ratio=0.25,
                equity_ratio=0.25,
                annual_rate=rate,
                sondertilgung_rate=0.0,
                refi_option=(None, None),
            )
            case = _build_case(fixed, combo, group_tag="collision_15pct_avoided")
            if case:
                cases.append(case)
    return cases


def _forced_exit_timing_pairs(fixed: dict) -> list[dict]:
    """
    Identical strategies, exit at year 9 vs year 11.
    Isolates the §23 speculation tax cliff effect.
    """
    cases = []
    base_combo = dict(
        usage="full_rental",
        rental_ratio=1.0,
        renovation_year_rel=5,
        renovation_amount=20_000,
        asset_split=10_000,
        land_ratio=0.25,
        equity_ratio=0.25,
        annual_rate=0.035,
        sondertilgung_rate=0.01,
        refi_option=(None, None),
    )
    for holding, tag in [(9, "exit_yr9"), (11, "exit_yr11")]:
        combo = {**base_combo, "holding_years": holding}
        case = _build_case(fixed, combo, group_tag=f"exit_timing_{tag}")
        if case:
            cases.append(case)
    return cases


def _forced_extreme_ltv(fixed: dict) -> list[dict]:
    """
    High LTV (low equity) vs low LTV (high equity) stress pairs.
    """
    cases = []
    for equity_ratio, tag in [(0.10, "high_ltv"), (0.50, "low_ltv")]:
        combo = dict(
            usage="full_rental",
            holding_years=10,
            rental_ratio=1.0,
            renovation_year_rel=5,
            renovation_amount=20_000,
            asset_split=10_000,
            land_ratio=0.25,
            equity_ratio=equity_ratio,
            annual_rate=0.040,
            sondertilgung_rate=0.0,
            refi_option=(None, None),
        )
        case = _build_case(fixed, combo, group_tag=f"stress_{tag}")
        if case:
            cases.append(case)
    return cases


# ---------------------------------------------------------------------------
# Main sampler function
# ---------------------------------------------------------------------------

def generate_cases(
    fixed_inputs: dict,
    include_forced_groups: bool = True,
    max_cases: Optional[int] = None,
) -> SamplerResult:
    """
    Generate all valid strategy combinations for a given property.

    Parameters
    ----------
    fixed_inputs : dict
        User-provided fixed parameters:
            purchase_price, state, purchase_year, personal_income,
            filing_status, market_rent_annual, building_type
    include_forced_groups : bool
        If True, prepend the analytically important groups (15% collision,
        exit timing pairs, LTV stress). These are always included regardless
        of max_cases.
    max_cases : int, optional
        Cap on the number of grid cases (after forced groups).
        None = no cap (full grid).

    Returns
    -------
    SamplerResult
        All valid cases with metadata.
    """
    required_fixed = {
        "purchase_price", "state", "purchase_year", "personal_income",
        "filing_status", "market_rent_annual", "building_type",
    }
    missing = required_fixed - set(fixed_inputs.keys())
    if missing:
        raise ValueError(f"fixed_inputs missing required keys: {missing}")

    all_cases: list[dict] = []
    group_counts: dict[str, int] = {}
    total_generated = 0
    total_invalid = 0

    # ---- Forced groups ----
    if include_forced_groups:
        forced = (
            _forced_15pct_collision(fixed_inputs)
            + _forced_15pct_avoided(fixed_inputs)
            + _forced_exit_timing_pairs(fixed_inputs)
            + _forced_extreme_ltv(fixed_inputs)
        )
        for c in forced:
            tag = c["_group_tag"]
            group_counts[tag] = group_counts.get(tag, 0) + 1
        all_cases.extend(forced)
        total_generated += len(forced)

    # ---- Full combinatorial grid ----
    dim_names  = [d[0] for d in _GRID_DIMENSIONS]
    dim_levels = [d[1] for d in _GRID_DIMENSIONS]

    grid_cases = []
    for values in itertools.product(*dim_levels):
        combo = dict(zip(dim_names, values))
        total_generated += 1

        case = _build_case(fixed_inputs, combo, group_tag="grid")
        if case is None:
            total_invalid += 1
            continue

        grid_cases.append(case)

        if max_cases is not None and len(grid_cases) >= max_cases:
            break

    group_counts["grid"] = len(grid_cases)
    all_cases.extend(grid_cases)

    # Deduplicate forced groups that coincidentally match grid entries
    # (keep forced tag — it has analytic value)
    seen: set[str] = set()
    deduped: list[dict] = []
    for c in all_cases:
        # Build a key from all non-metadata fields
        key_parts = {k: v for k, v in c.items() if not k.startswith("_")}
        key = str(sorted(key_parts.items()))
        if key not in seen:
            seen.add(key)
            deduped.append(c)

    total_valid = len(deduped)

    return SamplerResult(
        total_generated=total_generated,
        total_valid=total_valid,
        total_invalid=total_invalid,
        cases=deduped,
        group_counts=group_counts,
    )
