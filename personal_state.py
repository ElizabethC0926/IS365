"""
personal_state.py
-----------------
Core state representation for the Real Estate Investment MDP.

Two data classes:
  PropertyState  — snapshot of one property at a given point in time
  PersonalState  — complete financial picture of the investor

Key design decisions:
  - Phase 1: max 1 property (properties list has 0 or 1 element)
  - Phase 2: extend to max 3 properties with zero-padding
  - to_observation() returns a fixed-length float32 array for Policy Network
  - Discrete fields (state, building_type, filing_status, status) are
    one-hot encoded so the network never sees arbitrary integers
  - random() generates a legally valid initial state for Random Reset
  - All monetary values in EUR, rates as decimals, years as integers
"""

from __future__ import annotations
import random as _random
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_PROPERTIES_PHASE1 = 1
MAX_PROPERTIES_PHASE2 = 3
MAX_PROPERTIES         = MAX_PROPERTIES_PHASE1  # change to PHASE2 when ready

PROPERTY_STATUSES  = ["none", "owned_vacant", "owned_renting", "sold"]
BUILDING_TYPES     = ["standard", "neubau_post_2023", "denkmal"]
FILING_STATUSES    = ["single", "married"]
GERMAN_STATES      = [
    "Bayern", "Berlin", "Hamburg", "Bremen", "Sachsen",
    "Baden-Wuerttemberg", "Nordrhein-Westfalen", "Hessen",
    "Niedersachsen", "Rheinland-Pfalz", "Saarland", "Brandenburg",
    "Mecklenburg-Vorpommern", "Sachsen-Anhalt", "Thueringen",
    "Schleswig-Holstein",
]

# Observation vector dimensions
# PersonalState scalars: current_year, liquid_cash, annual_income, years_elapsed = 4
# PersonalState one-hot: filing_status (2) = 2
# PropertyState scalars per property: 10
# PropertyState one-hot per property: status(4) + building_type(3) + state(16) = 23
# Total per property: 10 + 23 = 33
# Phase 1 (1 property): 4 + 2 + 33 = 39
# Phase 2 (3 properties): 4 + 2 + 3*33 = 105

_PERSONAL_SCALAR_DIM  = 4
_PERSONAL_ONEHOT_DIM  = len(FILING_STATUSES)
_PROPERTY_SCALAR_DIM  = 10
_PROPERTY_ONEHOT_DIM  = (len(PROPERTY_STATUSES)
                          + len(BUILDING_TYPES)
                          + len(GERMAN_STATES))
_PROPERTY_TOTAL_DIM   = _PROPERTY_SCALAR_DIM + _PROPERTY_ONEHOT_DIM
OBS_DIM = (_PERSONAL_SCALAR_DIM
           + _PERSONAL_ONEHOT_DIM
           + MAX_PROPERTIES * _PROPERTY_TOTAL_DIM)


# ---------------------------------------------------------------------------
# PropertyState
# ---------------------------------------------------------------------------

@dataclass
class PropertyState:
    """
    Snapshot of one property at a given simulation year.

    Fields marked # cumulative are running totals that grow year by year
    and are needed for §23 speculation tax and 15% rule checks.
    """
    # Identity / classification
    status:           str    # "none" | "owned_vacant" | "owned_renting" | "sold"
    german_state:     str    # Bundesland (for Grunderwerbsteuer)
    building_type:    str    # "standard" | "neubau_post_2023" | "denkmal"

    # Purchase facts (fixed once bought)
    purchase_year:    int
    purchase_price:   float
    market_rent_annual: float    # benchmark for 66% rule

    # Loan
    current_loan_balance: float  # outstanding principal
    annual_rate:      float      # current interest rate

    # Cumulative trackers  # (reset to 0 at purchase, grow each year)
    cumulative_afa:         float   # total AfA claimed so far
    cumulative_renovation:  float   # total renovation spend (for 15% rule)

    # Rental income
    current_rent_annual: float   # 0 if not renting

    # Time
    years_owned: int             # years since purchase (for §23)

    def is_owned(self) -> bool:
        return self.status in ("owned_vacant", "owned_renting")

    def is_renting(self) -> bool:
        return self.status == "owned_renting"

    def to_vector(self) -> np.ndarray:
        """
        Serialize to fixed-length float32 array.
        Scalars are normalised to roughly [-1, 1] to help network training.
        One-hot fields are appended after scalars.
        """
        # ---- Scalars (10 values) ----
        scalars = np.array([
            self.purchase_year / 2030.0,                   # ≈ [0.99, 1.0]
            self.purchase_price / 1_000_000.0,             # ≈ [0, 1]
            self.market_rent_annual / 50_000.0,
            self.current_loan_balance / 1_000_000.0,
            self.annual_rate / 0.10,                       # max rate assumed 10%
            self.cumulative_afa / 100_000.0,
            self.cumulative_renovation / 200_000.0,
            self.current_rent_annual / 50_000.0,
            self.years_owned / 20.0,                       # max 20yr holding
            1.0 if self.status == "none" else 0.0,         # "empty" flag
        ], dtype=np.float32)

        # ---- One-hot encodings ----
        status_oh   = _one_hot(self.status, PROPERTY_STATUSES)
        btype_oh    = _one_hot(self.building_type, BUILDING_TYPES)
        state_oh    = _one_hot(self.german_state, GERMAN_STATES)

        return np.concatenate([scalars, status_oh, btype_oh, state_oh])

    @classmethod
    def empty(cls) -> "PropertyState":
        """A 'no property' placeholder for padding."""
        return cls(
            status="none",
            german_state=GERMAN_STATES[0],
            building_type="standard",
            purchase_year=2024,
            purchase_price=0.0,
            market_rent_annual=0.0,
            current_loan_balance=0.0,
            annual_rate=0.0,
            cumulative_afa=0.0,
            cumulative_renovation=0.0,
            current_rent_annual=0.0,
            years_owned=0,
        )

    @classmethod
    def random_owned(cls, purchase_year: int) -> "PropertyState":
        """Generate a random 'already owned' property for Random Reset."""
        price         = float(_random.choice([200_000, 300_000, 400_000, 500_000]))
        years_owned   = _random.randint(1, 8)
        loan_balance  = price * _random.uniform(0.3, 0.8)
        afa_rate      = 0.02
        building_val  = price * _random.uniform(0.6, 0.8)
        cum_afa       = building_val * afa_rate * years_owned
        is_renting    = _random.random() > 0.3
        market_rent   = price * _random.uniform(0.035, 0.055)

        return cls(
            status        = "owned_renting" if is_renting else "owned_vacant",
            german_state  = _random.choice(GERMAN_STATES[:6]),
            building_type = "standard",
            purchase_year = purchase_year - years_owned,
            purchase_price= price,
            market_rent_annual = market_rent,
            current_loan_balance = max(0.0, loan_balance),
            annual_rate   = _random.uniform(0.025, 0.045),
            cumulative_afa= cum_afa,
            cumulative_renovation = 0.0,
            current_rent_annual = market_rent * 0.95 if is_renting else 0.0,
            years_owned   = years_owned,
        )


# ---------------------------------------------------------------------------
# PersonalState
# ---------------------------------------------------------------------------

@dataclass
class PersonalState:
    """
    Complete financial picture of the investor at one point in time.

    This is the MDP State. env.reset() creates one of these;
    world_model.step() returns an updated copy after each action.
    """
    current_year:  int
    liquid_cash:   float          # available cash (EUR)
    annual_income: float          # salary/other income (EUR/yr)
    filing_status: str            # "single" | "married"
    properties:    list[PropertyState]
    years_elapsed: int            # steps taken in this episode

    # ------------------------------------------------------------------ #
    # Convenience properties
    # ------------------------------------------------------------------ #

    def owned_properties(self) -> list[PropertyState]:
        return [p for p in self.properties if p.is_owned()]

    def vacant_properties(self) -> list[PropertyState]:
        return [p for p in self.properties if p.status == "owned_vacant"]

    def renting_properties(self) -> list[PropertyState]:
        return [p for p in self.properties if p.is_renting()]

    def has_any_property(self) -> bool:
        return len(self.owned_properties()) > 0

    def total_loan_balance(self) -> float:
        return sum(p.current_loan_balance for p in self.owned_properties())

    def total_rental_income(self) -> float:
        return sum(p.current_rent_annual for p in self.renting_properties())

    def can_afford_purchase(
        self,
        purchase_price: float,
        ltv: float,
        nebenkosten_rate: float = 0.12,
    ) -> bool:
        """
        Conservative check: can investor cover equity + Nebenkosten?
        Nebenkosten ≈ 12% (GrESt 6% + Notar 1.5% + Makler 3.57% + buffer)
        """
        equity_needed = purchase_price * (1 - ltv)
        nebenkosten   = purchase_price * nebenkosten_rate
        return self.liquid_cash >= equity_needed + nebenkosten

    def property_count(self) -> int:
        return len(self.owned_properties())

    def at_property_limit(self) -> bool:
        return self.property_count() >= MAX_PROPERTIES

    # ------------------------------------------------------------------ #
    # Vectorization
    # ------------------------------------------------------------------ #

    def to_observation(self) -> np.ndarray:
        """
        Convert to fixed-length float32 observation vector.

        Layout:
          [personal_scalars (4)]
          [filing_status one-hot (2)]
          [property_0 vector (33)]
          [property_1 vector (33)]  ← zeros if no second property (Phase 2)
          ...
        Total: OBS_DIM (39 for Phase 1)
        """
        # Personal scalars
        personal = np.array([
            self.current_year / 2040.0,
            min(self.liquid_cash / 500_000.0, 2.0),
            min(self.annual_income / 200_000.0, 2.0),
            self.years_elapsed / 15.0,
        ], dtype=np.float32)

        filing_oh = _one_hot(self.filing_status, FILING_STATUSES)

        # Property vectors (pad with empty if fewer than MAX_PROPERTIES)
        owned = [p for p in self.properties if p.status != "none"]
        prop_vecs = []
        for i in range(MAX_PROPERTIES):
            if i < len(owned):
                prop_vecs.append(owned[i].to_vector())
            else:
                prop_vecs.append(PropertyState.empty().to_vector())

        return np.concatenate(
            [personal, filing_oh] + prop_vecs
        ).astype(np.float32)

    # ------------------------------------------------------------------ #
    # Factory methods
    # ------------------------------------------------------------------ #

    @classmethod
    def from_user_input(cls, user_dict: dict) -> "PersonalState":
        """
        Build a PersonalState from user-provided parameters.
        Used for Scenario B (existing owner) and Scenario A (no property).

        user_dict keys:
            current_year   : int
            liquid_cash    : float
            annual_income  : float
            filing_status  : "single" | "married"
            properties     : list[dict]  (each dict → PropertyState fields)
                             Empty list = Scenario A (no property yet)
        """
        props = []
        for pd in user_dict.get("properties", []):
            props.append(PropertyState(**pd))

        # Pad to MAX_PROPERTIES with empty slots
        while len(props) < MAX_PROPERTIES:
            props.append(PropertyState.empty())

        return cls(
            current_year  = user_dict["current_year"],
            liquid_cash   = float(user_dict["liquid_cash"]),
            annual_income = float(user_dict["annual_income"]),
            filing_status = user_dict.get("filing_status", "single"),
            properties    = props,
            years_elapsed = 0,
        )

    @classmethod
    def random(
        cls,
        current_year: int = 2025,
        scenario: str = "no_property",
    ) -> "PersonalState":
        """
        Generate a random legally valid initial state.
        Used by env.reset() for Random Reset during training.

        scenario:
            "no_property"   — Scenario A: investor has cash, no property
            "has_property"  — Scenario B: investor already owns a property
            "mixed"         — random choice between the two
        """
        if scenario == "mixed":
            scenario = _random.choice(["no_property", "has_property"])

        liquid_cash   = float(_random.choice([
            40_000, 60_000, 80_000, 100_000, 120_000, 150_000, 200_000
        ]))
        annual_income = float(_random.choice([
            40_000, 60_000, 80_000, 100_000, 120_000, 150_000
        ]))
        filing_status = _random.choice(FILING_STATUSES)

        if scenario == "no_property":
            props = [PropertyState.empty()] * MAX_PROPERTIES
        else:
            owned = PropertyState.random_owned(current_year)
            props = [owned] + [PropertyState.empty()] * (MAX_PROPERTIES - 1)

        return cls(
            current_year  = current_year,
            liquid_cash   = liquid_cash,
            annual_income = annual_income,
            filing_status = filing_status,
            properties    = props,
            years_elapsed = 0,
        )

    def copy(self) -> "PersonalState":
        """Deep copy — world_model always works on a copy, never mutates."""
        import copy
        return copy.deepcopy(self)

    def summary(self) -> str:
        """Human-readable one-liner for logging."""
        owned = self.owned_properties()
        rent  = sum(p.current_rent_annual for p in owned)
        loans = sum(p.current_loan_balance for p in owned)
        return (
            f"Yr{self.current_year} | "
            f"Cash €{self.liquid_cash:,.0f} | "
            f"Income €{self.annual_income:,.0f} | "
            f"Props {len(owned)} | "
            f"Rent €{rent:,.0f}/yr | "
            f"Loans €{loans:,.0f}"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _one_hot(value: str, categories: list[str]) -> np.ndarray:
    """Return a float32 one-hot vector. Unknown values → all zeros."""
    vec = np.zeros(len(categories), dtype=np.float32)
    if value in categories:
        vec[categories.index(value)] = 1.0
    return vec
