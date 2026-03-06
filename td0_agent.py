"""
td0_agent.py
============
TD(0) agent — baseline comparison for TD(λ).

Equivalent to TD(λ) with λ=0 (no eligibility traces).
Updates only use the immediate one-step TD error.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from td_lambda_agent import QNetwork


class TD0Agent:
    """
    Standard TD(0) Q-learning agent with neural network function approximation.

    No eligibility traces — credit is propagated only one step back.

    Parameters
    ----------
    obs_dim   : int    observation dimension
    n_actions : int    number of discrete actions
    gamma     : float  discount factor
    alpha     : float  learning rate
    epsilon   : float  ε-greedy exploration rate
    hidden    : int    hidden units in the MLP
    """

    def __init__(
        self,
        obs_dim:   int,
        n_actions: int,
        gamma:     float = 0.99,
        alpha:     float = 1e-3,
        epsilon:   float = 0.1,
        hidden:    int   = 128,
    ):
        self.gamma     = gamma
        self.epsilon   = epsilon
        self.n_actions = n_actions

        self.q_net     = QNetwork(obs_dim, n_actions, hidden)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=alpha)
        self.loss_fn   = nn.MSELoss()

        self.name = "TD(0)"

    # ------------------------------------------------------------------
    def reset_traces(self):
        """No-op: TD(0) has no eligibility traces."""
        pass

    # ------------------------------------------------------------------
    def select_action(self, obs: np.ndarray) -> int:
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        with torch.no_grad():
            q = self.q_net(self._to_tensor(obs))
        return int(q.argmax().item())

    # ------------------------------------------------------------------
    def update(
        self,
        obs:      np.ndarray,
        action:   int,
        reward:   float,
        next_obs: np.ndarray,
        done:     bool,
    ):
        """Standard one-step TD(0) update."""
        s      = self._to_tensor(obs)
        s_next = self._to_tensor(next_obs)

        q_values = self.q_net(s)
        q_sa     = q_values[action]

        with torch.no_grad():
            q_next  = self.q_net(s_next)
            v_next  = q_next.max() if not done else torch.tensor(0.0)
            target  = torch.tensor(
                reward + self.gamma * v_next.item(), dtype=torch.float32
            )

        loss = self.loss_fn(q_sa, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    # ------------------------------------------------------------------
    @staticmethod
    def _to_tensor(obs: np.ndarray) -> torch.Tensor:
        return torch.tensor(obs, dtype=torch.float32)

    def save(self, path: str):
        torch.save(self.q_net.state_dict(), path)

    def load(self, path: str):
        self.q_net.load_state_dict(torch.load(path))
