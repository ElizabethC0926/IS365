"""
reward.py
---------
Reward function for the Real Estate Investment MDP.

Takes the raw reward_components dict from world_model.step() and
combines them into a scalar reward the RL agent optimises.

Design principles:
  1. Staged development (see REWARD_STAGE in config):
       Stage 1 — only cashflow + exit (validate agent learns basics)
       Stage 2 — add FLAG penalties
       Stage 3 — add convergence bonus and shaping extras

  2. All lambda weights come from config so experiments need no code changes.

  3. Returns both the scalar reward AND a breakdown dict so Decision Log
     can attribute each episode's outcome to specific rule violations.

  4. Normalisation: rewards are divided by annual_income so the agent
     sees a consistent scale regardless of property price / income level.
     A reward of +1.0 ≈ "gained one year's salary in net cashflow".
"""

from __future__ import annotations
import json
import math
from pathlib import Path


# ---------------------------------------------------------------------------
# Default lambda weights (overridden by config.json if present)
# ---------------------------------------------------------------------------

DEFAULT_LAMBDAS = {
    # Stage 1 (always active)
    "annual_cashflow":      1.0,    # weight on net_cashflow / normaliser
    "exit_proceeds":        1.0,    # weight on net exit proceeds / normaliser

    # Stage 2 (active when stage >= 2)
    "flag_15pct_hit":      -0.30,   # one-time penalty when 15% rule fires
    "flag_rent_too_low":   -0.10,   # per-year penalty for below-66% rent
    "flag_tax_waste":      -0.20,   # per-year penalty for wasted deductions
    "flag_negative_cf":    -0.10,   # per-year penalty for negative cashflow

    # Stage 3 (active when stage >= 3)
    "convergence_bonus":    0.05,   # small bonus for early convergence
    "illegal_action":      -0.50,   # penalty for attempting illegal action
}

DEFAULT_CONFIG = {
    "reward_stage":    1,           # start with stage 1, raise in experiments
    "normaliser":      80_000.0,    # divide all monetary rewards by this
    "discount_gamma":  0.95,        # informational only; used by train.py
}


# ---------------------------------------------------------------------------
# RewardFunction
# ---------------------------------------------------------------------------

class RewardFunction:
    """
    Converts world_model reward_components into a scalar reward.

    Usage:
        rf = RewardFunction.from_config("config.json")
        reward, breakdown = rf.compute(reward_components, annual_income)
    """

    def __init__(self, lambdas: dict, stage: int, normaliser: float):
        self.lambdas    = {**DEFAULT_LAMBDAS, **lambdas}
        self.stage      = stage
        self.normaliser = max(normaliser, 1.0)

    # ------------------------------------------------------------------ #
    # Factory
    # ------------------------------------------------------------------ #

    @classmethod
    def from_config(cls, config_path: str = "config.json") -> "RewardFunction":
        """Load weights from config.json if it exists, else use defaults."""
        path = Path(config_path)
        cfg  = {}
        if path.exists():
            with open(path) as f:
                cfg = json.load(f)
        lambdas    = cfg.get("reward_lambdas", {})
        stage      = cfg.get("reward_stage",   DEFAULT_CONFIG["reward_stage"])
        normaliser = cfg.get("reward_normaliser", DEFAULT_CONFIG["normaliser"])
        return cls(lambdas, stage, normaliser)

    # ------------------------------------------------------------------ #
    # Compute
    # ------------------------------------------------------------------ #

    def compute(
        self,
        reward_components: dict,
        annual_income: float,
    ) -> tuple[float, dict]:
        """
        Compute scalar reward and per-component breakdown.

        Parameters
        ----------
        reward_components : dict
            Output of world_model.step() — contains:
              annual_net_cashflow, exit_net_proceeds,
              flag_15pct_hit, flag_rent_too_low,
              flag_tax_waste, flag_negative_cashflow,
              action_cash_delta (buy/sell/renovation cash flows)
        annual_income : float
            Used as fallback normaliser if normaliser == 0.

        Returns
        -------
        scalar_reward : float
            Single float the RL library optimises.
        breakdown : dict
            Per-component contributions (for Decision Log and debugging).
        """
        N = self.normaliser if self.normaliser > 0 else max(annual_income, 1.0)
        L = self.lambdas
        breakdown = {}

        # ── Stage 1: cashflow + exit ────────────────────────────────────
        cf   = reward_components.get("annual_net_cashflow", 0.0)
        exit = reward_components.get("exit_net_proceeds",   0.0)

        r_cf   = L["annual_cashflow"] * cf   / N
        r_exit = L["exit_proceeds"]   * exit / N

        breakdown["annual_cashflow"] = r_cf
        breakdown["exit_proceeds"]   = r_exit
        total = r_cf + r_exit

        # ── Stage 2: FLAG penalties ──────────────────────────────────────
        if self.stage >= 2:
            f15   = reward_components.get("flag_15pct_hit",         0.0)
            frent = reward_components.get("flag_rent_too_low",       0.0)
            fwaste= reward_components.get("flag_tax_waste",          0.0)
            fncf  = reward_components.get("flag_negative_cashflow",  0.0)

            r_f15   = L["flag_15pct_hit"]    * f15
            r_frent = L["flag_rent_too_low"] * frent
            r_fwaste= L["flag_tax_waste"]    * fwaste
            r_fncf  = L["flag_negative_cf"]  * fncf

            breakdown["flag_15pct_hit"]    = r_f15
            breakdown["flag_rent_too_low"] = r_frent
            breakdown["flag_tax_waste"]    = r_fwaste
            breakdown["flag_negative_cf"]  = r_fncf
            total += r_f15 + r_frent + r_fwaste + r_fncf

        # ── Stage 3: shaping extras ──────────────────────────────────────
        if self.stage >= 3:
            conv = reward_components.get("convergence_bonus", 0.0)
            ill  = reward_components.get("illegal_action",    0.0)

            r_conv = L["convergence_bonus"] * conv
            r_ill  = L["illegal_action"]    * ill

            breakdown["convergence_bonus"] = r_conv
            breakdown["illegal_action"]    = r_ill
            total += r_conv + r_ill

        # Clamp to avoid extreme outliers destabilising training
        total = float(max(-10.0, min(10.0, total)))
        breakdown["total"] = total

        return total, breakdown

    # ------------------------------------------------------------------ #
    # Terminal reward
    # ------------------------------------------------------------------ #

    def terminal(
        self,
        final_liquid_cash: float,
        initial_equity: float,
        annual_income: float,
    ) -> float:
        """
        Bonus/penalty at episode end (when max_steps reached without selling).
        Rewards the agent for building net worth vs. the starting equity.
        Normalised by annual_income.
        """
        N      = self.normaliser if self.normaliser > 0 else max(annual_income, 1.0)
        gain   = final_liquid_cash - initial_equity
        return float(max(-5.0, min(5.0, gain / N)))

    def summary(self) -> str:
        lines = [f"RewardFunction (stage={self.stage}, normaliser=€{self.normaliser:,.0f})"]
        for k, v in self.lambdas.items():
            active = ""
            if k in ("annual_cashflow", "exit_proceeds"):
                active = " [S1]"
            elif k.startswith("flag"):
                active = " [S2]" if self.stage >= 2 else " [S2 — inactive]"
            else:
                active = " [S3]" if self.stage >= 3 else " [S3 — inactive]"
            lines.append(f"  {k:<25}: {v:+.3f}{active}")
        return "\n".join(lines)
