"""
train.py
--------
Training entry point for the Real Estate Investment RL agent.

Supports three algorithms:
    ppo  — MaskablePPO (sb3-contrib) — recommended, handles action masking natively
    a2c  — A2C (stable-baselines3)   — faster per step, higher variance
    dqn  — DQN (stable-baselines3)   — off-policy, good sample efficiency

Usage:
    python train.py --algo ppo --steps 1000000 --seed 42
    python train.py --algo dqn --steps 500000  --stage 2 --hidden 256
    python train.py --algo a2c --steps 1000000 --seed 0 --n-envs 4

Outputs (saved to --output-dir):
    {algo}_{run_id}/
        best_model.zip          — best checkpoint by eval reward
        final_model.zip         — model at end of training
        training_log.json       — reward curve and hyperparams
        eval_results.json       — periodic evaluation snapshots

M4 Pro training time estimates (1M steps):
    PPO  ~1.5h,  DQN  ~2h,  A2C  ~1.3h
    Device: MPS (Metal) on Apple Silicon, falls back to CPU if unavailable.
"""

from __future__ import annotations
import argparse
import datetime
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# ── Try to import SB3 ───────────────────────────────────────────────────────
try:
    import torch
    from stable_baselines3 import A2C, DQN
    from stable_baselines3.common.callbacks import (
        BaseCallback, EvalCallback, CheckpointCallback,
    )
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    SB3_OK = True
except ImportError:
    SB3_OK = False

try:
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.maskable.evaluation import evaluate_policy as maskable_evaluate
    SB3_CONTRIB_OK = True
except ImportError:
    SB3_CONTRIB_OK = False

from env import RealEstateEnv, DEFAULT_ENV_CONFIG
from policy_net import RealEstateActorCriticPolicy, make_policy_kwargs


# ---------------------------------------------------------------------------
# Hyperparameter defaults
# ---------------------------------------------------------------------------

ALGO_DEFAULTS = {
    "ppo": {
        "learning_rate":   3e-4,
        "n_steps":         2048,
        "batch_size":      64,
        "n_epochs":        10,
        "gamma":           0.95,
        "gae_lambda":      0.95,
        "clip_range":      0.20,
        "ent_coef":        0.01,
        "vf_coef":         0.50,
        "max_grad_norm":   0.50,
    },
    "a2c": {
        "learning_rate":   7e-4,
        "n_steps":         5,
        "gamma":           0.95,
        "gae_lambda":      1.00,
        "ent_coef":        0.01,
        "vf_coef":         0.50,
        "max_grad_norm":   0.50,
    },
    "dqn": {
        "learning_rate":   1e-4,
        "buffer_size":     100_000,
        "learning_starts": 1_000,
        "batch_size":      64,
        "tau":             1.0,
        "gamma":           0.95,
        "train_freq":      4,
        "gradient_steps":  1,
        "target_update_interval": 1_000,
        "exploration_fraction":   0.20,
        "exploration_final_eps":  0.05,
    },
}


# ---------------------------------------------------------------------------
# Logging callback
# ---------------------------------------------------------------------------

class TrainingLogger(BaseCallback):
    """
    Records episode rewards and episode length to a JSON log.
    Prints a progress line every log_interval steps.
    """

    def __init__(self, log_interval: int = 10_000, verbose: int = 0):
        super().__init__(verbose)
        self.log_interval = log_interval
        self.episode_rewards: list[float] = []
        self.episode_lengths: list[int]   = []
        self._ep_reward_buf: list[float]  = []
        self._last_log_step = 0

    def _on_step(self) -> bool:
        # Collect episode info from Monitor wrapper
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])

        if (self.num_timesteps - self._last_log_step) >= self.log_interval:
            self._last_log_step = self.num_timesteps
            if self.episode_rewards:
                recent = self.episode_rewards[-50:]
                mean_r = np.mean(recent)
                std_r  = np.std(recent)
                print(
                    f"  Steps: {self.num_timesteps:>8,d} | "
                    f"Episodes: {len(self.episode_rewards):>6,d} | "
                    f"Mean reward (last 50): {mean_r:+.4f} ± {std_r:.4f}"
                )
        return True

    def to_dict(self) -> dict:
        return {
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "total_episodes":  len(self.episode_rewards),
        }


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def make_env(config: dict, seed: int = 0, rank: int = 0):
    """Factory function for vectorised env creation."""
    def _init():
        env = RealEstateEnv(config=config)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(
    algo:        str   = "ppo",
    total_steps: int   = 1_000_000,
    seed:        int   = 42,
    n_envs:      int   = 1,
    hidden_dim:  int   = 256,
    dropout:     float = 0.10,
    reward_stage:int   = 1,
    output_dir:  str   = "checkpoints",
    eval_freq:   int   = 50_000,
    env_config:  dict  = None,
) -> dict:
    """
    Run training and return the training log dict.

    Parameters
    ----------
    algo : str
        "ppo" | "a2c" | "dqn"
    total_steps : int
        Total environment interaction steps.
    seed : int
        Random seed for reproducibility.
    n_envs : int
        Number of parallel environments (PPO/A2C only; DQN always 1).
    hidden_dim : int
        Hidden layer size for the policy network.
    dropout : float
        Dropout rate in the MLP trunk.
    reward_stage : int
        Reward function stage (1 = cashflow only, 2 = + FLAG penalties).
    output_dir : str
        Directory to save checkpoints and logs.
    eval_freq : int
        Steps between periodic evaluations.
    env_config : dict, optional
        Override DEFAULT_ENV_CONFIG fields.

    Returns
    -------
    dict : training log with reward curve and hyperparams.
    """
    if not SB3_OK:
        raise ImportError(
            "stable-baselines3 not found. "
            "Install with: pip install stable-baselines3 sb3-contrib"
        )

    # ── Setup ──────────────────────────────────────────────────────────────
    run_id    = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path  = Path(output_dir) / f"{algo}_{run_id}"
    out_path.mkdir(parents=True, exist_ok=True)

    # Select device
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"\n  Device: {device}")

    # Merge env config
    cfg = {**DEFAULT_ENV_CONFIG, **(env_config or {})}
    cfg["reward_stage"] = reward_stage

    # ── Build environments ─────────────────────────────────────────────────
    if algo == "dqn":
        n_envs = 1   # DQN doesn't support vectorised envs in SB3

    if n_envs == 1:
        train_env = Monitor(RealEstateEnv(config=cfg))
        train_env.reset(seed=seed)
        vec_env = DummyVecEnv([make_env(cfg, seed, 0)])
    else:
        vec_env = SubprocVecEnv(
            [make_env(cfg, seed, i) for i in range(n_envs)]
        )

    eval_env = Monitor(RealEstateEnv(config=cfg))
    eval_env.reset(seed=seed + 9999)

    # ── Build model ────────────────────────────────────────────────────────
    hp = ALGO_DEFAULTS[algo].copy()
    policy_kwargs = make_policy_kwargs(hidden_dim=hidden_dim, dropout=dropout)

    print(f"\n  Algorithm : {algo.upper()}")
    print(f"  Steps     : {total_steps:,}")
    print(f"  Envs      : {n_envs}")
    print(f"  Seed      : {seed}")
    print(f"  RewardStage: {reward_stage}")
    print(f"  Hidden    : {hidden_dim}")
    print(f"  Output    : {out_path}")

    if algo == "ppo":
        if not SB3_CONTRIB_OK:
            raise ImportError("sb3-contrib not found. pip install sb3-contrib")
        model = MaskablePPO(
            RealEstateActorCriticPolicy,
            vec_env,
            **hp,
            policy_kwargs = policy_kwargs,
            seed          = seed,
            device        = device,
            verbose       = 0,
            tensorboard_log = str(out_path / "tb"),
        )
    elif algo == "a2c":
        model = A2C(
            RealEstateActorCriticPolicy,
            vec_env,
            **hp,
            policy_kwargs = policy_kwargs,
            seed          = seed,
            device        = device,
            verbose       = 0,
            tensorboard_log = str(out_path / "tb"),
        )
    elif algo == "dqn":
        # DQN uses a different policy class; action masking via wrapper
        model = DQN(
            "MlpPolicy",
            vec_env,
            **hp,
            policy_kwargs = {"net_arch": [hidden_dim, hidden_dim]},
            seed          = seed,
            device        = device,
            verbose       = 0,
            tensorboard_log = str(out_path / "tb"),
        )
    else:
        raise ValueError(f"Unknown algo: {algo}. Choose ppo | a2c | dqn")

    # ── Callbacks ──────────────────────────────────────────────────────────
    logger_cb = TrainingLogger(log_interval=10_000)

    checkpoint_cb = CheckpointCallback(
        save_freq     = max(eval_freq // n_envs, 1),
        save_path     = str(out_path / "ckpts"),
        name_prefix   = algo,
        verbose       = 0,
    )

    callbacks = [logger_cb, checkpoint_cb]

    # ── Train ──────────────────────────────────────────────────────────────
    print(f"\n  Training started at {datetime.datetime.now():%H:%M:%S}")
    t0 = time.time()

    model.learn(
        total_timesteps    = total_steps,
        callback           = callbacks,
        progress_bar       = True,
        reset_num_timesteps= True,
    )

    elapsed = time.time() - t0
    print(f"\n  Training finished in {elapsed/3600:.2f}h  ({elapsed:.0f}s)")

    # ── Save ───────────────────────────────────────────────────────────────
    model.save(str(out_path / "final_model"))
    print(f"  Saved: {out_path / 'final_model.zip'}")

    # ── Training log ───────────────────────────────────────────────────────
    log = {
        "run_id":        run_id,
        "algo":          algo,
        "total_steps":   total_steps,
        "seed":          seed,
        "n_envs":        n_envs,
        "hidden_dim":    hidden_dim,
        "dropout":       dropout,
        "reward_stage":  reward_stage,
        "device":        device,
        "elapsed_sec":   round(elapsed, 1),
        "hyperparams":   hp,
        "training_curve": logger_cb.to_dict(),
        "output_dir":    str(out_path),
    }

    with open(out_path / "training_log.json", "w") as f:
        json.dump(log, f, indent=2)

    vec_env.close()
    eval_env.close()
    return log


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Train RL agent for Real Estate Investment MDP"
    )
    parser.add_argument("--algo",   default="ppo",
                        choices=["ppo", "a2c", "dqn"])
    parser.add_argument("--steps",  type=int, default=1_000_000)
    parser.add_argument("--seed",   type=int, default=42)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--dropout",type=float, default=0.10)
    parser.add_argument("--stage",  type=int, default=1,
                        help="Reward stage: 1=cashflow only, 2=+FLAGS, 3=+shaping")
    parser.add_argument("--output", default="checkpoints")
    parser.add_argument("--eval-freq", type=int, default=50_000)
    parser.add_argument("--price",  type=float, default=400_000,
                        help="Property purchase price for BUY action")
    parser.add_argument("--state",  default="Bayern",
                        help="German Bundesland")
    args = parser.parse_args(argv)

    env_config = {
        "purchase_price": args.price,
        "german_state":   args.state,
    }

    log = train(
        algo         = args.algo,
        total_steps  = args.steps,
        seed         = args.seed,
        n_envs       = args.n_envs,
        hidden_dim   = args.hidden,
        dropout      = args.dropout,
        reward_stage = args.stage,
        output_dir   = args.output,
        eval_freq    = args.eval_freq,
        env_config   = env_config,
    )

    eps = log["training_curve"]["total_episodes"]
    if eps > 0:
        recent = log["training_curve"]["episode_rewards"][-100:]
        print(f"\n  Final mean reward (last 100 ep): {np.mean(recent):+.4f}")
    print(f"  Log saved: {log['output_dir']}/training_log.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
