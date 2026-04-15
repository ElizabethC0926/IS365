"""
env.py
------
Gymnasium environment for the Real Estate Investment MDP.

Implements the standard Gymnasium API:
    reset(seed, options) → (observation, info)
    step(action)         → (observation, reward, terminated, truncated, info)
    action_masks()       → np.ndarray[bool]  ← required by MaskablePPO

The environment wraps world_model.step() and handles:
  - Random Reset (training) vs. fixed initial state (inference/scenario B)
  - Action masking (delegated to action_mask.py)
  - Episode termination (max_steps or property sold)
  - Decision Log accumulation across the episode
  - Reward computation (delegated to reward.py)

Configuration is passed as a dict at construction time so multiple
parallel environments can share the same config without file I/O.
"""

from __future__ import annotations
import copy
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from personal_state import PersonalState, PropertyState, OBS_DIM, GERMAN_STATES
from action_space import N_ACTIONS, get_action, ActionType, actions_of_type
from action_mask import compute_mask
from world_model import step as world_step
from reward import RewardFunction
from tax_engine import TaxEngine


# ---------------------------------------------------------------------------
# Default environment configuration
# ---------------------------------------------------------------------------

DEFAULT_ENV_CONFIG = {
    # Episode length
    "max_steps":           15,          # years per episode
    "early_stop_window":   5,           # steps with no improvement before early stop
    "early_stop_threshold":0.01,        # min improvement fraction to not trigger

    # Property parameters (used when BUY action is taken)
    "purchase_price":      400_000.0,
    "german_state":        "Bayern",
    "building_type":       "standard",

    # Initial state distribution (for Random Reset)
    "scenario":            "mixed",     # "no_property" | "has_property" | "mixed"
    "start_year":          2025,

    # Reward
    "reward_stage":        1,
    "reward_normaliser":   80_000.0,

    # Tax params file
    "tax_params_path":     "tax_params.json",
}


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class RealEstateEnv(gym.Env):
    """
    Single-asset real estate investment environment.

    Observation space : Box(OBS_DIM,) float32
    Action space      : Discrete(N_ACTIONS)
    Action masking    : action_masks() → bool array (for MaskablePPO)
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        config: dict | None = None,
        initial_state: PersonalState | None = None,
        render_mode: str | None = None,
    ):
        super().__init__()

        self.config = {**DEFAULT_ENV_CONFIG, **(config or {})}
        self._fixed_initial_state = initial_state   # None = use Random Reset
        self.render_mode = render_mode

        # Gymnasium spaces
        self.observation_space = spaces.Box(
            low=-5.0, high=5.0, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(N_ACTIONS)

        # Shared objects (created once, reused across episodes)
        self._tax_engine = TaxEngine(self.config["tax_params_path"])
        self._reward_fn  = RewardFunction(
            lambdas    = self.config.get("reward_lambdas", {}),
            stage      = self.config["reward_stage"],
            normaliser = self.config["reward_normaliser"],
        )

        # Episode state (reset each episode)
        self._state:          PersonalState | None = None
        self._step_count:     int = 0
        self._episode_log:    list[dict] = []
        self._initial_equity: float = 0.0
        self._recent_rewards: list[float] = []

    # ------------------------------------------------------------------ #
    # Gymnasium API
    # ------------------------------------------------------------------ #

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        # Allow caller to inject a specific state (Scenario B / evaluation)
        injected = (options or {}).get("initial_state", self._fixed_initial_state)

        if injected is not None:
            self._state = copy.deepcopy(injected)
        else:
            # Random Reset — set numpy seed so reproducible when seed given
            if seed is not None:
                import random
                random.seed(seed)
            self._state = PersonalState.random(
                current_year = self.config["start_year"],
                scenario     = self.config["scenario"],
            )

        # Ensure property slots have the configured german_state
        for i, prop in enumerate(self._state.properties):
            if prop.status == "none":
                p = copy.copy(prop)
                p.german_state = self.config["german_state"]
                self._state.properties[i] = p

        self._step_count     = 0
        self._episode_log    = []
        self._initial_equity = self._state.liquid_cash
        self._recent_rewards = []

        obs  = self._state.to_observation()
        info = {"state_summary": self._state.summary()}
        return obs, info

    def step(
        self, action_idx: int
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self._state is None:
            raise RuntimeError("Call reset() before step()")

        action = get_action(action_idx)

        # Detect illegal action (shouldn't happen with MaskablePPO but be safe)
        mask = compute_mask(self._state)
        if not mask[action_idx]:
            # Substitute DO_NOTHING and apply illegal-action signal
            action = get_action(0)
            illegal_signal = 1.0
        else:
            illegal_signal = 0.0

        # World model step
        new_state, reward_comps, info = world_step(
            state                  = self._state,
            action                 = action,
            tax_engine             = self._tax_engine,
            purchase_price         = self.config["purchase_price"],
            target_property_index  = 0,
        )
        reward_comps["illegal_action"] = illegal_signal

        # Compute scalar reward
        scalar_reward, breakdown = self._reward_fn.compute(
            reward_comps, self._state.annual_income
        )
        self._recent_rewards.append(scalar_reward)

        self._state      = new_state
        self._step_count += 1

        # Log this step for Decision Log
        self._episode_log.append({
            "step":       self._step_count,
            "year":       info["year"],
            "action":     info["action"],
            "flags":      info["flags"],
            "reward":     scalar_reward,
            "breakdown":  breakdown,
            "totals":     info["totals"],
        })

        # Termination conditions
        sold       = action.action_type == ActionType.SELL_PROPERTY
        max_reached= self._step_count >= self.config["max_steps"]
        early_stop = self._check_early_stop()

        terminated = sold or max_reached
        truncated  = early_stop and not terminated

        # Terminal reward (only on natural termination)
        if terminated or truncated:
            term_r = self._reward_fn.terminal(
                final_liquid_cash = self._state.liquid_cash,
                initial_equity    = self._initial_equity,
                annual_income     = self._state.annual_income,
            )
            scalar_reward += term_r
            self._episode_log[-1]["terminal_reward"] = term_r

        obs = self._state.to_observation()
        info.update({
            "step":          self._step_count,
            "action_taken":  action.action_type.name,
            "reward_breakdown": breakdown,
            "episode_log":   self._episode_log if (terminated or truncated) else [],
            "state_summary": self._state.summary(),
        })

        if self.render_mode == "human":
            self.render()

        return obs, scalar_reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        """
        Required by sb3-contrib MaskablePPO.
        Returns boolean array of shape [N_ACTIONS].
        """
        if self._state is None:
            return np.ones(N_ACTIONS, dtype=bool)
        return compute_mask(self._state)

    def render(self) -> str | None:
        if self._state is None:
            return None
        lines = [
            f"\n── Step {self._step_count} / {self.config['max_steps']} ──",
            self._state.summary(),
        ]
        mask = compute_mask(self._state)
        legal = [get_action(i).action_type.name for i in range(N_ACTIONS) if mask[i]]
        lines.append(f"Legal actions: {set(legal)}")
        out = "\n".join(lines)
        if self.render_mode == "human":
            print(out)
        return out

    def close(self):
        pass

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _check_early_stop(self) -> bool:
        """
        Trigger early stop if reward has been flat for early_stop_window steps.
        Only activates after at least 2× the window has elapsed.
        """
        window = self.config["early_stop_window"]
        thresh = self.config["early_stop_threshold"]
        if len(self._recent_rewards) < window * 2:
            return False
        recent = self._recent_rewards[-window:]
        older  = self._recent_rewards[-window * 2:-window]
        if not older:
            return False
        improvement = (sum(recent) - sum(older)) / (abs(sum(older)) + 1e-8)
        return improvement < thresh

    @property
    def current_state(self) -> PersonalState | None:
        return self._state

    @property
    def episode_log(self) -> list[dict]:
        return self._episode_log

    def get_episode_summary(self) -> dict:
        """Return summary stats for the completed episode."""
        if not self._episode_log:
            return {}
        total_reward = sum(e["reward"] for e in self._episode_log)
        flags_hit    = {k: any(e["flags"].get(k) for e in self._episode_log)
                        for k in ["FLAG_15_PERCENT_HIT", "FLAG_RENT_TOO_LOW",
                                  "FLAG_TAX_WASTE", "FLAG_NEGATIVE_CASHFLOW"]}
        return {
            "steps":           self._step_count,
            "total_reward":    round(total_reward, 4),
            "final_cash":      self._state.liquid_cash if self._state else 0,
            "initial_equity":  self._initial_equity,
            "net_gain":        (self._state.liquid_cash - self._initial_equity)
                               if self._state else 0,
            "flags_hit":       flags_hit,
            "any_flag":        any(flags_hit.values()),
        }
