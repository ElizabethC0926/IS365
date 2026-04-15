"""
validators.py
-------------
Parameter validation before any simulation runs.

Design:
  - Returns ALL errors at once (not fail-fast) so the user sees
    every problem in one pass.
  - Each rule is a standalone function returning list[str] so rules
    can be tested independently.
  - No business calculation here — only constraint checking.
  - Called by sampler.py (to filter bad samples) and main.py (to
    validate user input before running).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    valid: bool
    errors: list[str]
    warnings: list[str]

    def __bool__(self):
        return self.valid

    def summary(self) -> str:
        lines = []
        if self.errors:
            lines.append(f"{len(self.errors)} error(s):")
            for e in self.errors:
                lines.append(f"  ✗  {e}")
        if self.warnings:
            lines.append(f"{len(self.warnings)} warning(s):")
            for w in self.warnings:
                lines.append(f"  ⚠  {w}")
        if self.valid and not self.warnings:
            lines.append("✓  All parameters valid.")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Individual rule functions
# ---------------------------------------------------------------------------

def _check_purchase_price(p: dict) -> tuple[list, list]:
    errors, warnings = [], []
    v = p.get("purchase_price", 0)
    if not isinstance(v, (int, float)) or v <= 0:
        errors.append(f"purchase_price must be > 0, got {v!r}")
    elif v < 50_000:
        warnings.append(f"purchase_price €{v:,.0f} seems very low.")
    elif v > 5_000_000:
        warnings.append(f"purchase_price €{v:,.0f} is unusually high.")
    return errors, warnings


def _check_state(p: dict) -> tuple[list, list]:
    valid_states = {
        "Bayern", "Berlin", "Hamburg", "Bremen", "Sachsen",
        "Baden-Wuerttemberg", "Nordrhein-Westfalen", "Hessen",
        "Niedersachsen", "Rheinland-Pfalz", "Saarland", "Brandenburg",
        "Mecklenburg-Vorpommern", "Sachsen-Anhalt", "Thueringen",
        "Schleswig-Holstein",
    }
    v = p.get("state", "")
    errors, warnings = [], []
    if v not in valid_states:
        errors.append(
            f"state '{v}' not recognised. Valid: {sorted(valid_states)}"
        )
    return errors, warnings


def _check_years(p: dict) -> tuple[list, list]:
    errors, warnings = [], []
    purchase_year = p.get("purchase_year", 0)
    holding_years = p.get("holding_years", 0)

    if not isinstance(purchase_year, int) or purchase_year < 2000:
        errors.append(f"purchase_year must be int >= 2000, got {purchase_year!r}")

    if not isinstance(holding_years, int) or holding_years < 1:
        errors.append(f"holding_years must be int >= 1, got {holding_years!r}")
    elif holding_years > 30:
        warnings.append(f"holding_years={holding_years} is unusually long.")

    return errors, warnings


def _check_income(p: dict) -> tuple[list, list]:
    errors, warnings = [], []
    v = p.get("personal_income", -1)
    if not isinstance(v, (int, float)) or v < 0:
        errors.append(f"personal_income must be >= 0, got {v!r}")
    elif v == 0:
        warnings.append("personal_income=0: tax shield calculations will be limited.")

    fs = p.get("filing_status", "")
    if fs not in ("single", "married"):
        errors.append(f"filing_status must be 'single' or 'married', got {fs!r}")

    return errors, warnings


def _check_usage_and_rent(p: dict) -> tuple[list, list]:
    errors, warnings = [], []
    usage        = p.get("usage", "")
    rental_ratio = p.get("rental_ratio", -1)
    market_rent  = p.get("market_rent_annual", 0)

    valid_usage = {"full_rental", "partial", "own_use"}
    if usage not in valid_usage:
        errors.append(f"usage must be one of {valid_usage}, got {usage!r}")

    if not isinstance(rental_ratio, (int, float)):
        errors.append(f"rental_ratio must be a number, got {rental_ratio!r}")
    elif not (0.0 <= rental_ratio <= 1.5):
        errors.append(f"rental_ratio {rental_ratio} out of range [0.0, 1.5]")
    elif rental_ratio < 0.5 and usage != "own_use":
        warnings.append(
            f"rental_ratio={rental_ratio:.0%} is below 50% — "
            "no Werbungskosten deduction will apply (§21 Abs.2 EStG)."
        )

    # Consistency: own_use should have rental_ratio=0
    if usage == "own_use" and rental_ratio > 0:
        warnings.append(
            f"usage='own_use' but rental_ratio={rental_ratio} > 0. "
            "rental_ratio will be ignored (no rental income for own_use)."
        )

    # Consistency: full_rental should not have zero rent
    if usage == "full_rental" and rental_ratio == 0:
        errors.append("usage='full_rental' requires rental_ratio > 0.")

    if not isinstance(market_rent, (int, float)) or market_rent <= 0:
        errors.append(f"market_rent_annual must be > 0, got {market_rent!r}")

    return errors, warnings


def _check_renovation(p: dict) -> tuple[list, list]:
    errors, warnings = [], []
    reno_year   = p.get("renovation_year", 0)
    reno_amount = p.get("renovation_amount", -1)
    holding     = p.get("holding_years", 0)
    price       = p.get("purchase_price", 0)

    if not isinstance(reno_year, int) or reno_year < 1:
        errors.append(f"renovation_year must be int >= 1, got {reno_year!r}")
    elif isinstance(holding, int) and reno_year > holding:
        errors.append(
            f"renovation_year={reno_year} exceeds holding_years={holding}."
        )

    if not isinstance(reno_amount, (int, float)) or reno_amount < 0:
        errors.append(f"renovation_amount must be >= 0, got {reno_amount!r}")
    elif price > 0 and reno_amount > price * 0.5:
        warnings.append(
            f"renovation_amount €{reno_amount:,.0f} exceeds 50% of purchase price. "
            "Unusual — double-check."
        )

    return errors, warnings


def _check_asset_split_and_land(p: dict) -> tuple[list, list]:
    errors, warnings = [], []
    asset_split = p.get("asset_split", -1)
    land_ratio  = p.get("land_ratio", -1)
    price       = p.get("purchase_price", 0)

    if not isinstance(asset_split, (int, float)) or asset_split < 0:
        errors.append(f"asset_split must be >= 0, got {asset_split!r}")
    elif price > 0 and asset_split > price * 0.15:
        warnings.append(
            f"asset_split €{asset_split:,.0f} is >15% of purchase price. "
            "Finanzamt may challenge this split."
        )

    if not isinstance(land_ratio, (int, float)):
        errors.append(f"land_ratio must be a number, got {land_ratio!r}")
    elif not (0.05 <= land_ratio <= 0.70):
        errors.append(
            f"land_ratio={land_ratio} out of realistic range [0.05, 0.70]."
        )
    elif land_ratio > 0.50:
        warnings.append(
            f"land_ratio={land_ratio:.0%} is high. "
            "Verify with local Bodenrichtwert data."
        )

    # land + movables must not exceed purchase price
    if price > 0 and isinstance(asset_split, (int, float)) and isinstance(land_ratio, (int, float)):
        land_value = price * land_ratio
        if land_value + asset_split >= price:
            errors.append(
                f"land_value (€{land_value:,.0f}) + asset_split (€{asset_split:,.0f}) "
                f">= purchase_price (€{price:,.0f}). Building value would be zero or negative."
            )

    return errors, warnings


def _check_financing(p: dict) -> tuple[list, list]:
    errors, warnings = [], []
    equity       = p.get("equity_amount", -1)
    rate         = p.get("annual_rate", -1)
    sonder       = p.get("sondertilgung_rate", -1)
    refi_year    = p.get("refi_year", None)
    refi_rate    = p.get("refi_rate", None)
    price        = p.get("purchase_price", 0)
    holding      = p.get("holding_years", 0)

    # Equity
    if not isinstance(equity, (int, float)) or equity < 0:
        errors.append(f"equity_amount must be >= 0, got {equity!r}")
    elif price > 0 and equity > price * 1.2:
        warnings.append(
            f"equity_amount €{equity:,.0f} exceeds purchase price. "
            "Full-cash purchase — loan will be zero."
        )

    # Interest rate
    if not isinstance(rate, (int, float)) or rate < 0:
        errors.append(f"annual_rate must be >= 0, got {rate!r}")
    elif rate > 0.15:
        warnings.append(f"annual_rate={rate:.1%} is very high (>15%). Check input.")
    elif rate == 0:
        warnings.append("annual_rate=0: interest-free loan assumed.")

    # Sondertilgung
    if not isinstance(sonder, (int, float)) or sonder < 0:
        errors.append(f"sondertilgung_rate must be >= 0, got {sonder!r}")
    elif sonder > 0.10:
        errors.append(
            f"sondertilgung_rate={sonder:.0%} exceeds 10%. "
            "German contracts typically cap at 5%."
        )

    # Refi consistency
    if refi_year is not None and refi_rate is None:
        errors.append("refi_rate must be provided when refi_year is set.")
    if refi_rate is not None and refi_year is None:
        errors.append("refi_year must be provided when refi_rate is set.")
    if refi_year is not None:
        if not isinstance(refi_year, int) or refi_year < 1:
            errors.append(f"refi_year must be int >= 1, got {refi_year!r}")
        elif isinstance(holding, int) and refi_year >= holding:
            errors.append(
                f"refi_year={refi_year} must be < holding_years={holding}."
            )
    if refi_rate is not None:
        if not isinstance(refi_rate, (int, float)) or refi_rate < 0:
            errors.append(f"refi_rate must be >= 0, got {refi_rate!r}")

    return errors, warnings


def _check_building_type(p: dict) -> tuple[list, list]:
    valid = {"standard", "neubau_post_2023", "denkmal"}
    v = p.get("building_type", "")
    errors = [] if v in valid else [
        f"building_type must be one of {valid}, got {v!r}"
    ]
    return errors, []


# ---------------------------------------------------------------------------
# Main validation function
# ---------------------------------------------------------------------------

# Required keys and their expected Python types
_REQUIRED_KEYS: dict[str, tuple] = {
    "purchase_price":     (int, float),
    "state":              (str,),
    "purchase_year":      (int,),
    "personal_income":    (int, float),
    "filing_status":      (str,),
    "market_rent_annual": (int, float),
    "building_type":      (str,),
    "usage":              (str,),
    "holding_years":      (int,),
    "rental_ratio":       (int, float),
    "renovation_year":    (int,),
    "renovation_amount":  (int, float),
    "asset_split":        (int, float),
    "land_ratio":         (int, float),
    "equity_amount":      (int, float),
    "annual_rate":        (int, float),
    "sondertilgung_rate": (int, float),
    "refi_year":          (int, type(None)),
    "refi_rate":          (int, float, type(None)),
}

_RULES = [
    _check_purchase_price,
    _check_state,
    _check_years,
    _check_income,
    _check_usage_and_rent,
    _check_renovation,
    _check_asset_split_and_land,
    _check_financing,
    _check_building_type,
]


def validate(params: dict) -> ValidationResult:
    """
    Validate a parameter dict for PropertyCase.

    Checks:
      1. All required keys are present and have correct types.
      2. Each business rule (ranges, consistency constraints).

    Returns ValidationResult with all errors and warnings collected.
    Never raises — caller decides what to do with errors.
    """
    all_errors: list[str] = []
    all_warnings: list[str] = []

    # ---- Step 1: required keys and types ----
    for key, expected_types in _REQUIRED_KEYS.items():
        if key not in params:
            all_errors.append(f"Missing required parameter: '{key}'")
            continue
        val = params[key]
        if not isinstance(val, expected_types):
            type_names = " | ".join(t.__name__ for t in expected_types)
            all_errors.append(
                f"'{key}' expected {type_names}, got {type(val).__name__} ({val!r})"
            )

    # ---- Step 2: business rules ----
    # Only run rules if keys are present (avoids cascading errors)
    if not any("Missing" in e for e in all_errors):
        for rule_fn in _RULES:
            errs, warns = rule_fn(params)
            all_errors.extend(errs)
            all_warnings.extend(warns)

    return ValidationResult(
        valid=len(all_errors) == 0,
        errors=all_errors,
        warnings=all_warnings,
    )


def validate_batch(param_list: list[dict]) -> list[tuple[int, ValidationResult]]:
    """
    Validate a list of parameter dicts.
    Returns only the invalid ones as (index, result) pairs.
    """
    invalid = []
    for i, p in enumerate(param_list):
        result = validate(p)
        if not result.valid:
            invalid.append((i, result))
    return invalid
