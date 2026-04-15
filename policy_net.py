"""
policy_net.py
-------------
Actor-Critic MLP backbone for the Real Estate Investment MDP.

Architecture:
  - Shared feature extractor (MLP trunk)
  - Separate heads for policy (actor) and value (critic)
  - Dropout regularisation (helps with sparse reward environments)

This module defines the network architecture only.
Training is handled by stable-baselines3 algorithms in train.py,
which accept a custom policy via the `policy_kwargs` argument.

Why MLP (not Transformer)?
  - State dimension is small (39 for Phase 1)
  - No sequential dependency within a single observation
  - Transformer overhead is not justified at this scale
  - Paper can include an ablation: MLP vs Transformer backbone

For stable-baselines3 integration, we subclass ActorCriticPolicy
to inject our custom network. The SB3 framework then handles the
training loop, advantage estimation, and optimisation.
"""

from __future__ import annotations
from typing import Callable, Optional

import torch
import torch.nn as nn
import numpy as np
from gymnasium import spaces

# stable-baselines3 imports (installed via pip install stable-baselines3)
try:
    from stable_baselines3.common.policies import ActorCriticPolicy
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    # Provide stubs so the file is importable even without SB3
    class BaseFeaturesExtractor:  # type: ignore
        pass
    class ActorCriticPolicy:  # type: ignore
        pass

from personal_state import OBS_DIM


# ---------------------------------------------------------------------------
# Feature extractor (shared trunk)
# ---------------------------------------------------------------------------

class RealEstateFeaturesExtractor(BaseFeaturesExtractor):
    """
    Two-layer MLP feature extractor shared between actor and critic heads.

    The observation vector already contains normalised scalars and one-hot
    encodings, so no embedding layer is needed.

    Architecture:
        Linear(obs_dim → hidden) → LayerNorm → ReLU → Dropout
        Linear(hidden → hidden)  → LayerNorm → ReLU → Dropout
        Output: hidden-dim feature vector

    LayerNorm instead of BatchNorm because:
      - Works with batch size 1 (used during rollout collection)
      - More stable in RL settings where batch statistics shift
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        hidden_dim: int = 256,
        dropout: float = 0.10,
    ):
        super().__init__(observation_space, features_dim=hidden_dim)

        obs_dim = int(np.prod(observation_space.shape))

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations)


# ---------------------------------------------------------------------------
# Custom Actor-Critic Policy (SB3 compatible)
# ---------------------------------------------------------------------------

class RealEstateActorCriticPolicy(ActorCriticPolicy):
    """
    Thin wrapper around SB3's ActorCriticPolicy that injects
    RealEstateFeaturesExtractor as the shared trunk.

    Usage (in train.py):
        from stable_baselines3 import PPO
        model = PPO(
            RealEstateActorCriticPolicy,
            env,
            policy_kwargs={"hidden_dim": 256, "dropout": 0.1},
        )

    For MaskablePPO (sb3-contrib), the same policy class works because
    MaskablePPO applies masking in the action selection step, not inside
    the policy network itself.
    """

    def __init__(self, *args, hidden_dim: int = 256, dropout: float = 0.10, **kwargs):
        self._hidden_dim = hidden_dim
        self._dropout    = dropout
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = RealEstateFeaturesExtractor(
            self.observation_space,
            hidden_dim = self._hidden_dim,
            dropout    = self._dropout,
        )
        # SB3 expects these attributes after _build_mlp_extractor
        self.latent_dim_pi = self._hidden_dim
        self.latent_dim_vf = self._hidden_dim

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        return features

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return features


# ---------------------------------------------------------------------------
# Standalone network (for testing without SB3)
# ---------------------------------------------------------------------------

class StandaloneActorCritic(nn.Module):
    """
    Self-contained Actor-Critic network.
    Used for unit tests and for DQN (which has a different SB3 policy API).

    Actor  → logits over N_ACTIONS (apply action mask before softmax)
    Critic → scalar value estimate
    """

    def __init__(
        self,
        obs_dim:    int = OBS_DIM,
        n_actions:  int = 34,
        hidden_dim: int = 256,
        dropout:    float = 0.10,
    ):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.actor  = nn.Linear(hidden_dim, n_actions)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        obs: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        obs : Tensor [batch, obs_dim]
        action_mask : BoolTensor [batch, n_actions] or None
            True = legal action. Illegal actions get logit = -1e9.

        Returns
        -------
        logits : Tensor [batch, n_actions]
        value  : Tensor [batch, 1]
        """
        features = self.trunk(obs)
        logits   = self.actor(features)
        value    = self.critic(features)

        if action_mask is not None:
            # Set illegal action logits to -inf before any softmax
            logits = logits.masked_fill(~action_mask, -1e9)

        return logits, value

    def get_action(
        self,
        obs: torch.Tensor,
        action_mask: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample (or argmax) an action.

        Returns
        -------
        actions     : Tensor [batch]
        log_probs   : Tensor [batch]
        values      : Tensor [batch, 1]
        """
        logits, value = self.forward(obs, action_mask)
        dist = torch.distributions.Categorical(logits=logits)
        if deterministic:
            actions = logits.argmax(dim=-1)
        else:
            actions = dist.sample()
        log_probs = dist.log_prob(actions)
        return actions, log_probs, value

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Used during PPO update step."""
        logits, value = self.forward(obs, action_mask)
        dist          = torch.distributions.Categorical(logits=logits)
        log_probs     = dist.log_prob(actions)
        entropy       = dist.entropy()
        return log_probs, value, entropy


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def make_policy_kwargs(hidden_dim: int = 256, dropout: float = 0.10) -> dict:
    """
    Returns the policy_kwargs dict for SB3 PPO/A2C constructor.

    Usage:
        model = MaskablePPO(
            RealEstateActorCriticPolicy,
            env,
            **make_policy_kwargs(hidden_dim=256),
        )
    """
    return {
        "hidden_dim": hidden_dim,
        "dropout":    dropout,
    }
