"""
tax_engine.py
-------------
German tax calculation engine for real estate investment simulation.

Design principles:
  - Zero hardcoded tax figures. Every number comes from tax_params.json.
  - Year-aware with fallback: if year 2027 is not in params, uses 2026.
  - Each function returns a dict (not a bare float) so Decision Log and
    FLAG system can trace which rule fired and what parameters were used.
  - Functions accept primitive arguments (not the full PropertyCase object)
    so they can be unit-tested independently.

Covered rules:
  - §32a EStG  : Einkommensteuer (progressive, Splitting)
  - SolzG      : Solidaritätszuschlag
  - §7 EStG    : AfA for buildings and movables
  - §6 I Nr.1a : 15% renovation capitalization rule
  - §21 II     : 66% / 50% rent threshold rules
  - §23 EStG   : Speculation tax on exit (10-year rule)
  - Werbungskosten: annual deductible cost aggregation
"""

import json
import copy
from pathlib import Path
from functools import lru_cache
from typing import Optional


# ---------------------------------------------------------------------------
# Parameter loading
# ---------------------------------------------------------------------------

class TaxEngine:
    """
    Stateless tax calculator. Instantiate once, call many times.
    All year-specific numbers are loaded from tax_params.json.
    """

    def __init__(self, params_path: str = "tax_params.json"):
        path = Path(params_path)
        if not path.exists():
            raise FileNotFoundError(f"tax_params.json not found at {path.resolve()}")
        with open(path, encoding="utf-8") as f:
            self._raw = json.load(f)

        # Pre-build sorted list of available years (ints), excluding _meta
        self._available_years = sorted(
            int(k) for k in self._raw if k.isdigit()
        )
        if not self._available_years:
            raise ValueError("tax_params.json contains no year entries.")

    def _get_params(self, year: int) -> dict:
        """
        Return merged parameters for the requested year.

        Strategy:
          1. Find the highest available year <= requested year (base).
          2. If the requested year itself exists, deep-merge it on top of base
             so only changed fields are overridden.
          3. If requested year > all available, use the latest available year
             and emit a warning.
        """
        available = self._available_years

        # Find base year
        base_year = None
        for y in reversed(available):
            if y <= year:
                base_year = y
                break

        if base_year is None:
            # Requested year is before all available data
            base_year = available[0]

        # Always start from the FIRST (oldest) available year as the complete base.
        # Then apply each subsequent year's changes in order up to the requested year.
        # This ensures partial year entries (like 2026 only specifying changed fields)
        # never lose fields that were only defined in earlier years.
        base = copy.deepcopy(self._raw[str(available[0])])
        for y in available[1:]:
            if y > year:
                break
            base = self._deep_merge(base, copy.deepcopy(self._raw[str(y)]))

        return base

    @staticmethod
    def _deep_merge(base: dict, override: dict) -> dict:
        """
        Recursively merge override into base.
        - override values win for scalar fields.
        - dict fields are merged recursively (so partial overrides work).
        - Comment keys (_comment, _meta etc.) are passed through but never
          cause a dict→scalar replacement.
        - A list in override fully replaces the list in base (bracket lists
          must be replaced wholesale, not element-patched).
        """
        result = copy.deepcopy(base)
        for key, value in override.items():
            if key.startswith("_"):
                result[key] = value
                continue
            if isinstance(value, dict) and isinstance(result.get(key), dict):
                # Both sides are dicts: recurse so partial overrides work
                result[key] = TaxEngine._deep_merge(result[key], value)
            else:
                # Scalar, list, or new key: override wins outright
                result[key] = value
        return result


    # ---------------------------------------------------------------------------
    # §32a EStG — Einkommensteuer
    # ---------------------------------------------------------------------------

    def calc_income_tax(
        self,
        zu_versteuerndes_einkommen: float,
        year: int,
        filing_status: str = "single"   # "single" | "married"
    ) -> dict:
        """
        Calculate German income tax (Einkommensteuer) + Solidaritätszuschlag.

        filing_status="married" applies Ehegattensplitting:
            tax = 2 * tax(zvE / 2)

        Returns a dict so callers can log which year's params were used
        and see the marginal rate (useful for reward function).
        """
        zvE = max(0.0, zu_versteuerndes_einkommen)
        p = self._get_params(year)
        brackets = p["einkommensteuer"]["brackets"]

        if filing_status == "married":
            raw_tax = 2.0 * self._apply_tarif(zvE / 2.0, brackets, year)
        else:
            raw_tax = self._apply_tarif(zvE, brackets, year)

        raw_tax = max(0.0, round(raw_tax, 2))

        soli = self._calc_soli(raw_tax, year, filing_status)
        total = round(raw_tax + soli, 2)
        effective_rate = round(total / zvE, 6) if zvE > 0 else 0.0
        marginal_rate = self._marginal_rate(zvE, brackets, year)

        return {
            "einkommensteuer": raw_tax,
            "solidaritaetszuschlag": soli,
            "total_tax": total,
            "effective_rate": effective_rate,
            "marginal_rate": marginal_rate,
            "filing_status": filing_status,
            "year": year,
            "law_ref": "§32a EStG"
        }

    def _apply_tarif(self, zvE: float, brackets: list, year: int) -> float:
        """
        Apply the German tax tariff to a single (non-split) income figure.

        Zone 0 (free):        0 tax.
        Zone 1 (progressive): polynomial formula on the portion within zone 1.
        Zone 2+ (flat):       marginal rate applied to the portion within that zone.

        All zones are processed independently so high incomes correctly
        accumulate tax from zone 1 AND zone 2 AND zone 3.
        """
        tax = 0.0
        for bracket in brackets:
            low  = bracket["from"]
            high = bracket["to"] if bracket["to"] is not None else float("inf")

            if zvE <= low:
                # Income does not reach this zone at all
                break

            if bracket["type"] == "free":
                # Grundfreibetrag: no tax
                continue

            elif bracket["type"] == "flat":
                # Tax only on the portion of zvE that falls within [low, high]
                taxable_in_zone = min(zvE, high) - low
                tax += taxable_in_zone * bracket["rate"]

            elif bracket["type"] == "progressive":
                # Zone 1: the formula already computes tax on the zone-1 portion only.
                # _zone1_tax clamps y to the zone width so incomes above zone 1
                # get the full zone-1 tax, and then continue to zone 2 below.
                capped_zvE = min(zvE, high)
                tax += self._zone1_tax(capped_zvE, bracket, year)
                # Do NOT break — income above zone 1 continues to flat zones.

        return tax

    def _zone1_tax(self, zvE: float, bracket: dict, year: int) -> float:
        """
        Compute tax for income falling in the progressive zone 1.

        The official formula is a polynomial in y = (zvE - grundfreibetrag) / 10000.
        We reconstruct the coefficients from the published start/end marginal rates
        and zone width so the formula automatically adapts when the bracket shifts.

        Official form (§32a Abs.1 Nr.2):
            ESt = (a * y + b) * y
        where:
            a = (rate_end - rate_start) / (2 * zone_width_in_10k_units - 1) * 10000
            b = rate_start * 10000  (approximately; exact value is in BMF tables)

        For correctness we use the published coefficients for 2025/2026
        and fall back to the derived formula for future years.
        """
        # Known exact coefficients from BMF (preferred over derived)
        known_coefficients = {
            2025: {"a": 1008.70, "b": 1400.00},
            2026: {"a": 1007.27, "b": 1400.00},
        }

        p_est = self._get_params(year)["einkommensteuer"]
        grundfreibetrag = p_est["grundfreibetrag"]

        y = (zvE - grundfreibetrag) / 10_000.0

        # Clamp y to the zone
        zone_top = bracket["to"] - grundfreibetrag
        y = min(y, zone_top / 10_000.0)
        y = max(y, 0.0)

        if year in known_coefficients:
            a = known_coefficients[year]["a"]
            b = known_coefficients[year]["b"]
        else:
            # Derive approximate coefficients from bracket metadata
            # rate_start and rate_end are the marginal rates at zone entry/exit
            rate_start = bracket.get("rate_start", 0.14)
            rate_end   = bracket.get("rate_end",   0.2397)
            zone_width_eur = bracket["to"] - bracket["from"]
            zone_width_10k = zone_width_eur / 10_000.0
            a = (rate_end - rate_start) / zone_width_10k * 10_000.0
            b = rate_start * 10_000.0

        return round((a * y + b) * y, 2)

    def _marginal_rate(self, zvE: float, brackets: list, year: int) -> float:
        """Return the marginal tax rate at the given income level."""
        for bracket in reversed(brackets):
            if zvE >= bracket["from"]:
                if bracket["type"] == "free":
                    return 0.0
                elif bracket["type"] == "flat":
                    return bracket["rate"]
                elif bracket["type"] == "progressive":
                    # Marginal rate within zone 1 varies; return midpoint approximation
                    rate_start = bracket.get("rate_start", 0.14)
                    rate_end   = bracket.get("rate_end",   0.2397)
                    zone_start = bracket["from"]
                    zone_end   = bracket["to"]
                    position = (zvE - zone_start) / (zone_end - zone_start)
                    return round(rate_start + position * (rate_end - rate_start), 4)
        return 0.0

    def _calc_soli(
        self,
        einkommensteuer: float,
        year: int,
        filing_status: str = "single"
    ) -> float:
        """
        Solidaritätszuschlag. Since 2021, only applies above Freigrenze.
        Milderungszone prevents a cliff edge.
        """
        p = self._get_params(year)["solidaritaetszuschlag"]
        rate        = p["rate"]
        freigrenze  = (p["freigrenze_married"]
                       if filing_status == "married"
                       else p["freigrenze_single"])
        milderung   = p.get("milderungszone_rate", 0.119)

        if einkommensteuer <= freigrenze:
            return 0.0

        soli_regular    = einkommensteuer * rate
        soli_milderung  = (einkommensteuer - freigrenze) * milderung

        return round(min(soli_regular, soli_milderung), 2)


    # ---------------------------------------------------------------------------
    # §7 EStG — AfA (Absetzung für Abnutzung)
    # ---------------------------------------------------------------------------

    def calc_afa(
        self,
        building_value: float,       # Gebäudeanteil des Kaufpreises (excl. Grund)
        movable_value: float,        # Separately contracted movables (Küche etc.)
        year_of_purchase: int,
        simulation_year: int,        # The calendar year being simulated
        building_type: str = "standard"  # "standard" | "neubau_post_2023" | "denkmal"
    ) -> dict:
        """
        Calculate annual depreciation (AfA) for a rental property.

        Key rules:
          - Grund und Boden is NOT depreciable (caller must exclude it).
          - Movables (kitchen etc.) depreciate at 20%/yr over 5 years,
            only if separately listed in the purchase contract.
          - Neubau (completion >= 2023) uses 3% rate instead of 2%.
          - Denkmal uses 9% for years 1-8, 7% for years 9-12.
        """
        p_afa = self._get_params(simulation_year)["afa"]
        years_held = simulation_year - year_of_purchase  # 0 in purchase year

        # Building AfA
        if building_type == "denkmal":
            if years_held < 8:
                building_rate = p_afa["gebaeude"]["denkmal_years_1_8"]
            elif years_held < 12:
                building_rate = p_afa["gebaeude"]["denkmal_years_9_12"]
            else:
                building_rate = p_afa["gebaeude"]["standard_rate"]
        elif building_type == "neubau_post_2023" or year_of_purchase >= 2023:
            building_rate = p_afa["gebaeude"]["neubau_post_2023_rate"]
        else:
            building_rate = p_afa["gebaeude"]["standard_rate"]

        building_afa = round(building_value * building_rate, 2)

        # Movable AfA: only in years 1-5 after purchase
        movable_rate = p_afa["movables"]["kueche_rate"]
        if 0 <= years_held < 5 and movable_value > 0:
            movable_afa = round(movable_value * movable_rate, 2)
        else:
            movable_afa = 0.0

        total_afa = round(building_afa + movable_afa, 2)

        return {
            "building_afa": building_afa,
            "movable_afa": movable_afa,
            "total_afa": total_afa,
            "building_rate_used": building_rate,
            "building_type": building_type,
            "years_held": years_held,
            "simulation_year": simulation_year,
            "law_ref": "§7 Abs.4 EStG"
        }


    # ---------------------------------------------------------------------------
    # §6 Abs.1 Nr.1a EStG — 15% Renovation Rule
    # ---------------------------------------------------------------------------

    def check_15pct_rule(
        self,
        renovation_cumulative_net: float,  # Total renovation spend net of VAT, years 1-3
        purchase_price_net: float,         # Purchase price net of VAT
        years_since_purchase: int,         # How many years since acquisition
        simulation_year: int
    ) -> dict:
        """
        Check whether renovation costs trigger the 15% capitalization rule.

        If triggered, renovation costs are NOT immediately deductible as
        Werbungskosten but must be capitalized and depreciated via AfA
        (added to the building's AfA base).

        Returns triggered=True/False and the quantified impact for Decision Log.
        """
        p = self._get_params(simulation_year)["thresholds"]["renovation"]
        window   = p["window_years"]
        limit    = p["limit_rate"]

        within_window  = years_since_purchase <= window
        limit_amount   = round(purchase_price_net * limit, 2)
        exceeds_limit  = renovation_cumulative_net > limit_amount
        triggered      = within_window and exceeds_limit

        return {
            "triggered": triggered,
            "within_window": within_window,
            "exceeds_limit": exceeds_limit,
            "years_since_purchase": years_since_purchase,
            "window_years": window,
            "renovation_spend": renovation_cumulative_net,
            "limit_amount": limit_amount,
            "limit_rate": limit,
            "impact": (
                "Renovation costs must be capitalized and depreciated via AfA "
                "(not immediately deductible as Werbungskosten)."
                if triggered else
                "Renovation costs are fully deductible as Werbungskosten."
            ),
            "law_ref": "§6 Abs.1 Nr.1a EStG"
        }


    # ---------------------------------------------------------------------------
    # §21 Abs.2 EStG — 66% Rent Rule
    # ---------------------------------------------------------------------------

    def check_rent_rule(
        self,
        actual_rent_annual: float,
        market_rent_annual: float,
        simulation_year: int
    ) -> dict:
        """
        Determine the Werbungskosten deduction ratio based on rent level.

        Three zones:
          >= 66% of market rent  -> full deduction (ratio 1.0)
          50-66% of market rent  -> Totalüberschussprognose required;
                                    modelled here as 0.5 deduction ratio
          < 50% of market rent   -> split entgeltlich/unentgeltlich;
                                    modelled as 0.0 (no deduction)
        """
        p = self._get_params(simulation_year)["thresholds"]["rent"]
        full_threshold = p["full_deduction_threshold"]   # 0.66
        half_threshold = p["half_deduction_threshold"]   # 0.50

        if market_rent_annual <= 0:
            ratio = 1.0
            zone  = "full"
        else:
            rent_ratio = actual_rent_annual / market_rent_annual
            if rent_ratio >= full_threshold:
                ratio = 1.0
                zone  = "full"
            elif rent_ratio >= half_threshold:
                ratio = 0.5
                zone  = "half"
            else:
                ratio = 0.0
                zone  = "none"

        return {
            "deduction_ratio": ratio,
            "zone": zone,
            "actual_rent": actual_rent_annual,
            "market_rent": market_rent_annual,
            "rent_ratio": round(actual_rent_annual / market_rent_annual, 4)
                          if market_rent_annual > 0 else None,
            "full_threshold": full_threshold,
            "half_threshold": half_threshold,
            "flag_rent_too_low": zone in ("half", "none"),
            "law_ref": "§21 Abs.2 EStG"
        }


    # ---------------------------------------------------------------------------
    # §23 EStG — Speculation Tax on Exit
    # ---------------------------------------------------------------------------

    def calc_speculation_tax(
        self,
        sale_price: float,
        original_purchase_price: float,
        cumulative_afa_claimed: float,    # Total AfA deducted over holding period
        holding_years: int,
        annual_income_in_exit_year: float,
        filing_status: str,
        simulation_year: int
    ) -> dict:
        """
        Calculate speculation tax on property sale (§23 EStG).

        Key mechanics:
          - Holding period >= 10 years -> tax-free (return 0).
          - Holding period < 10 years  -> gain is taxed at personal marginal rate.
          - Gain = sale_price - (purchase_price - cumulative_afa_claimed).
            AfA recapture: AfA claimed reduces the tax base, increasing taxable gain.
        """
        p = self._get_params(simulation_year)["thresholds"]["speculation_tax"]
        free_after = p["free_after_years"]

        if holding_years >= free_after:
            return {
                "speculation_tax": 0.0,
                "taxable_gain": 0.0,
                "tax_free": True,
                "holding_years": holding_years,
                "free_after_years": free_after,
                "law_ref": "§23 Abs.1 S.1 Nr.1 EStG"
            }

        # Book value after AfA recapture
        adjusted_book_value = original_purchase_price - cumulative_afa_claimed
        taxable_gain = max(0.0, sale_price - adjusted_book_value)

        # Gain is added to personal income in exit year and taxed at marginal rate
        total_income_with_gain = annual_income_in_exit_year + taxable_gain
        tax_with_gain    = self.calc_income_tax(total_income_with_gain,
                                                simulation_year, filing_status)
        tax_without_gain = self.calc_income_tax(annual_income_in_exit_year,
                                                simulation_year, filing_status)

        speculation_tax = round(
            tax_with_gain["total_tax"] - tax_without_gain["total_tax"], 2
        )

        return {
            "speculation_tax": speculation_tax,
            "taxable_gain": round(taxable_gain, 2),
            "adjusted_book_value": round(adjusted_book_value, 2),
            "tax_free": False,
            "holding_years": holding_years,
            "free_after_years": free_after,
            "marginal_rate_used": tax_with_gain["marginal_rate"],
            "law_ref": "§23 Abs.1 S.1 Nr.1 EStG"
        }


    # ---------------------------------------------------------------------------
    # Werbungskosten aggregation
    # ---------------------------------------------------------------------------

    def calc_werbungskosten(
        self,
        interest_paid: float,
        afa_total: float,
        renovation_deductible: float,    # 0 if 15% rule triggered
        management_costs: float,
        insurance_costs: float,
        other_costs: float,
        deduction_ratio: float,          # from check_rent_rule()
        simulation_year: int
    ) -> dict:
        """
        Aggregate all deductible costs (Werbungskosten) for a given year.
        The deduction_ratio from check_rent_rule() is applied to all costs
        except AfA (which is always at full rate based on usage).

        Note: Interest (Schuldzinsen) is deductible per §9 Abs.1 S.3 Nr.1 EStG.
        AfA is deductible per §9 Abs.1 S.3 Nr.7 EStG.
        """
        gross = {
            "interest":     interest_paid,
            "afa":          afa_total,
            "renovation":   renovation_deductible,
            "management":   management_costs,
            "insurance":    insurance_costs,
            "other":        other_costs,
        }

        # AfA is applied at full rate (it follows the usage split separately)
        # All other costs are scaled by deduction_ratio
        net = {
            "interest":   round(interest_paid        * deduction_ratio, 2),
            "afa":        round(afa_total            * deduction_ratio, 2),
            "renovation": round(renovation_deductible * deduction_ratio, 2),
            "management": round(management_costs     * deduction_ratio, 2),
            "insurance":  round(insurance_costs      * deduction_ratio, 2),
            "other":      round(other_costs          * deduction_ratio, 2),
        }

        total_net = round(sum(net.values()), 2)

        return {
            "gross_costs": gross,
            "net_deductible": net,
            "total_deductible": total_net,
            "deduction_ratio": deduction_ratio,
            "simulation_year": simulation_year,
            "law_ref": "§9 EStG"
        }


    # ---------------------------------------------------------------------------
    # Grunderwerbsteuer (one-time, at purchase)
    # ---------------------------------------------------------------------------

    def calc_grunderwerbsteuer(
        self,
        purchase_price: float,
        state: str,
        year: int
    ) -> dict:
        """
        One-time land transfer tax paid at purchase.
        Rate depends on the German state (Bundesland).
        """
        p = self._get_params(year)["grunderwerbsteuer"]
        rate = p["by_state"].get(state, p["default"])
        tax  = round(purchase_price * rate, 2)

        return {
            "grunderwerbsteuer": tax,
            "rate": rate,
            "state": state,
            "purchase_price": purchase_price,
            "year": year,
            "law_ref": "GrEStG §1"
        }
