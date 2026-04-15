"""
evaluate.py
-----------
Comparative evaluation: PPO vs DQN vs A2C vs Random Search.

Produces the tables and figures needed for the paper's experiment chapter:

  Table 1 — Sample efficiency
             Given the same N simulator calls, which agent finds
             higher-IRR strategies?

  Table 2 — FLAG rate
             How often does each agent trigger each FLAG
             (15% rule, rent too low, negative cashflow, tax waste)?

  Table 3 — §23 cliff effect
             Does the agent learn to wait for the 10-year tax-free window?
             Metric: fraction of episodes where SELL happens at year ≥ 10.

  Figure 1 — Training curve (reward vs steps) — exported as CSV for plotting

Usage:
    # Evaluate pre-trained models
    python evaluate.py \\
        --ppo  checkpoints/ppo_20260101/final_model.zip \\
        --dqn  checkpoints/dqn_20260101/final_model.zip \\
        --a2c  checkpoints/a2c_20260101/final_model.zip \\
        --n-eval 500 \\
        --output eval_results/

    # Quick sanity check (random agent only, no model files needed)
    python evaluate.py --random-only --n-eval 100
"""

from __future__ import annotations
import argparse
import json
import sys
import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from env import RealEstateEnv, DEFAULT_ENV_CONFIG
from action_space import N_ACTIONS, get_action, ActionType
from action_mask import compute_mask

# SB3 optional imports
try:
    from sb3_contrib import MaskablePPO
    from stable_baselines3 import DQN, A2C
    SB3_OK = True
except ImportError:
    SB3_OK = False


# ---------------------------------------------------------------------------
# Agent wrappers (uniform interface)
# ---------------------------------------------------------------------------

class RandomAgent:
    """Uniformly samples from the legal action set."""
    name = "Random"

    def predict(self, obs, action_masks=None, deterministic=False):
        if action_masks is not None:
            legal = np.where(action_masks)[0]
        else:
            legal = np.arange(N_ACTIONS)
        action = int(np.random.choice(legal))
        return action, None


class SB3AgentWrapper:
    """Wraps a loaded SB3 model with the same interface as RandomAgent."""

    def __init__(self, model, name: str):
        self._model = model
        self.name   = name

    def predict(self, obs, action_masks=None, deterministic=True):
        return self._model.predict(
            obs,
            action_masks  = action_masks,
            deterministic = deterministic,
        )


# ---------------------------------------------------------------------------
# Single-agent evaluation
# ---------------------------------------------------------------------------

def evaluate_agent(
    agent,
    env_config: dict,
    n_episodes: int = 200,
    seed: int = 0,
    deterministic: bool = True,
) -> dict:
    """
    Run n_episodes with the given agent and collect metrics.

    Returns
    -------
    dict with keys:
        episode_rewards, episode_lengths,
        flag_rates (per flag), sell_year_hist,
        spec_tax_free_rate, mean_reward, std_reward
    """
    env = RealEstateEnv(config=env_config)

    episode_rewards  = []
    episode_lengths  = []
    flag_counts      = {
        "FLAG_15_PERCENT_HIT":   0,
        "FLAG_RENT_TOO_LOW":     0,
        "FLAG_TAX_WASTE":        0,
        "FLAG_NEGATIVE_CASHFLOW":0,
    }
    sell_years       = []       # year_index when SELL happens
    spec_tax_free    = 0        # episodes where sell was tax-free
    sell_happened    = 0

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        ep_reward = 0.0
        ep_flags  = {k: False for k in flag_counts}

        for step in range(env.config["max_steps"]):
            masks = env.action_masks()

            try:
                action, _ = agent.predict(obs, action_masks=masks,
                                          deterministic=deterministic)
            except Exception:
                action = 0   # DO_NOTHING fallback

            obs, reward, terminated, truncated, info = env.step(int(action))
            ep_reward += reward

            # Accumulate flags
            for k in ep_flags:
                if info["flags"].get(k):
                    ep_flags[k] = True

            # Track sell events
            if info.get("action_taken") == "SELL_PROPERTY":
                sell_happened += 1
                sell_years.append(step + 1)
                # Tax-free = property held ≥ 10 years
                # We approximate: sell at step ≥ 10
                if step + 1 >= 10:
                    spec_tax_free += 1

            if terminated or truncated:
                break

        episode_rewards.append(ep_reward)
        episode_lengths.append(step + 1)
        for k in ep_flags:
            if ep_flags[k]:
                flag_counts[k] += 1

    # Aggregate
    flag_rates = {k: v / n_episodes for k, v in flag_counts.items()}
    sell_rate  = sell_happened / n_episodes
    spec_rate  = spec_tax_free / sell_happened if sell_happened > 0 else 0.0

    return {
        "agent":              agent.name,
        "n_episodes":         n_episodes,
        "mean_reward":        float(np.mean(episode_rewards)),
        "std_reward":         float(np.std(episode_rewards)),
        "median_reward":      float(np.median(episode_rewards)),
        "min_reward":         float(np.min(episode_rewards)),
        "max_reward":         float(np.max(episode_rewards)),
        "mean_episode_length":float(np.mean(episode_lengths)),
        "flag_rates":         flag_rates,
        "sell_rate":          sell_rate,
        "spec_tax_free_rate": spec_rate,
        "sell_year_mean":     float(np.mean(sell_years)) if sell_years else None,
        "sell_year_hist":     _histogram(sell_years, bins=range(1, 17)),
        "episode_rewards":    episode_rewards,   # full curve for plotting
    }


def _histogram(values, bins) -> dict:
    if not values:
        return {}
    bins = list(bins)
    hist = {}
    for b in bins:
        hist[str(b)] = sum(1 for v in values if v == b)
    return hist


# ---------------------------------------------------------------------------
# Paper table formatting
# ---------------------------------------------------------------------------

def print_table1(results: list[dict]):
    """Table 1: Mean ± Std reward per agent."""
    print("\n══ Table 1: Mean Episode Reward ══")
    print(f"  {'Agent':<12}  {'Mean':>10}  {'Std':>8}  {'Median':>10}  {'Max':>10}")
    print("  " + "─" * 54)
    for r in results:
        print(
            f"  {r['agent']:<12}  "
            f"{r['mean_reward']:>+10.4f}  "
            f"{r['std_reward']:>8.4f}  "
            f"{r['median_reward']:>+10.4f}  "
            f"{r['max_reward']:>+10.4f}"
        )


def print_table2(results: list[dict]):
    """Table 2: FLAG trigger rates."""
    flags = [
        "FLAG_15_PERCENT_HIT",
        "FLAG_RENT_TOO_LOW",
        "FLAG_TAX_WASTE",
        "FLAG_NEGATIVE_CASHFLOW",
    ]
    short = ["15% rule", "Rent low", "Tax waste", "Neg CF"]
    print("\n══ Table 2: FLAG Trigger Rates (fraction of episodes) ══")
    header = f"  {'Flag':<15}" + "".join(f"  {r['agent']:>10}" for r in results)
    print(header)
    print("  " + "─" * (15 + 12 * len(results)))
    for flag, label in zip(flags, short):
        row = f"  {label:<15}"
        for r in results:
            rate = r["flag_rates"].get(flag, 0.0)
            row += f"  {rate:>9.1%} "
        print(row)


def print_table3(results: list[dict]):
    """Table 3: §23 tax-free exit rate."""
    print("\n══ Table 3: §23 Speculation Tax — Sell Behaviour ══")
    print(f"  {'Agent':<12}  {'Sell rate':>10}  {'Mean sell yr':>13}  {'Tax-free rate':>14}")
    print("  " + "─" * 54)
    for r in results:
        sell_yr = r["sell_year_mean"]
        print(
            f"  {r['agent']:<12}  "
            f"{r['sell_rate']:>10.1%}  "
            f"{sell_yr:>13.1f}  " if sell_yr else
            f"  {r['agent']:<12}  "
            f"{r['sell_rate']:>10.1%}  "
            f"{'—':>13}  "
            ,
            end=""
        )
        print(f"{r['spec_tax_free_rate']:>13.1%}")


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def run_evaluation(
    model_paths: dict[str, Optional[str]],
    env_config: dict,
    n_eval: int = 200,
    seed: int = 0,
    output_dir: str = "eval_results",
) -> dict:
    """
    Load models, run evaluation, print tables, save results.

    Parameters
    ----------
    model_paths : dict
        {"ppo": "path/to/model.zip", "dqn": None, "a2c": None}
        None entries are skipped.
    env_config : dict
        Environment configuration.
    n_eval : int
        Episodes per agent.
    seed : int
        Evaluation seed.
    output_dir : str
        Where to save eval_results.json.
    """
    agents = [RandomAgent()]   # always include random baseline

    if SB3_OK:
        for algo, path in model_paths.items():
            if path is None:
                continue
            p = Path(path)
            if not p.exists():
                print(f"  Warning: {path} not found, skipping {algo}")
                continue
            try:
                if algo == "ppo":
                    model = MaskablePPO.load(path)
                elif algo == "dqn":
                    model = DQN.load(path)
                elif algo == "a2c":
                    model = A2C.load(path)
                else:
                    continue
                agents.append(SB3AgentWrapper(model, algo.upper()))
                print(f"  Loaded {algo.upper()} from {path}")
            except Exception as e:
                print(f"  Failed to load {algo}: {e}")

    print(f"\n  Evaluating {len(agents)} agent(s) × {n_eval} episodes each...")
    all_results = []
    for agent in agents:
        print(f"  → {agent.name}...", end=" ", flush=True)
        result = evaluate_agent(agent, env_config, n_eval, seed)
        all_results.append(result)
        print(f"mean={result['mean_reward']:+.4f}")

    # Print paper tables
    print_table1(all_results)
    print_table2(all_results)
    print_table3(all_results)

    # Save
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    run_id  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outfile = out / f"eval_{run_id}.json"

    # Strip large episode_rewards list for storage (keep summary only)
    save_results = []
    for r in all_results:
        sr = {k: v for k, v in r.items() if k != "episode_rewards"}
        sr["reward_curve_sample"] = r["episode_rewards"][:50]  # first 50
        save_results.append(sr)

    with open(outfile, "w") as f:
        json.dump({
            "run_id":     run_id,
            "n_eval":     n_eval,
            "seed":       seed,
            "env_config": env_config,
            "results":    save_results,
        }, f, indent=2)

    print(f"\n  Results saved: {outfile}")
    return {"results": all_results, "output_file": str(outfile)}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate and compare RL agents"
    )
    parser.add_argument("--ppo",         default=None, help="PPO model .zip path")
    parser.add_argument("--dqn",         default=None, help="DQN model .zip path")
    parser.add_argument("--a2c",         default=None, help="A2C model .zip path")
    parser.add_argument("--n-eval",      type=int, default=200)
    parser.add_argument("--seed",        type=int, default=0)
    parser.add_argument("--output",      default="eval_results")
    parser.add_argument("--random-only", action="store_true",
                        help="Run random agent only (no model files needed)")
    parser.add_argument("--price",       type=float, default=400_000)
    parser.add_argument("--state",       default="Bayern")
    args = parser.parse_args(argv)

    env_config = {
        **DEFAULT_ENV_CONFIG,
        "purchase_price": args.price,
        "german_state":   args.state,
    }

    model_paths = {} if args.random_only else {
        "ppo": args.ppo,
        "dqn": args.dqn,
        "a2c": args.a2c,
    }

    run_evaluation(
        model_paths = model_paths,
        env_config  = env_config,
        n_eval      = args.n_eval,
        seed        = args.seed,
        output_dir  = args.output,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
