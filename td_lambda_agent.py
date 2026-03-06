"""
td_lambda_agent.py
==================
TD(λ) agent with neural network function approximation for
adaptive portfolio rebalancing.

Algorithm:
  - Policy: ε-greedy over Q-values predicted by an MLP.
  - Value function: Q(s, a) approximated by a 2-layer neural network.
  - Credit assignment: accumulating eligibility traces for each weight.
  - Update: w ← w + α · δ_t · e_t  (applied every step).

Parameters:
  lambda_val : float   λ decay factor for eligibility traces
  gamma      : float   discount factor
  alpha      : float   learning rate
  epsilon    : float   exploration rate (ε-greedy)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ---------------------------------------------------------------------------
# Neural Network — shared architecture across all three agents
# ---------------------------------------------------------------------------
class QNetwork(nn.Module):
    """2-layer MLP that maps (state) → Q-values for all actions."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# TD(λ) Agent
# ---------------------------------------------------------------------------
class TDLambdaAgent:
    """
    TD(λ) agent with accumulating eligibility traces and Q-learning.

    Parameters
    ----------
    obs_dim   : int    dimension of the observation vector
    n_actions : int    number of discrete actions
    lambda_val: float  λ ∈ [0,1] — trace decay
    gamma     : float  discount factor
    alpha     : float  learning rate
    epsilon   : float  ε-greedy exploration rate
    hidden    : int    hidden units in the MLP
    """

    def __init__(
        self,
        obs_dim:    int,
        n_actions:  int,
        lambda_val: float = 0.7,
        gamma:      float = 0.99,
        alpha:      float = 1e-3,
        epsilon:    float = 0.1,
        hidden:     int   = 128,
    ):
        self.lambda_val = lambda_val
        self.gamma      = gamma
        self.epsilon    = epsilon
        self.n_actions  = n_actions

        self.q_net   = QNetwork(obs_dim, n_actions, hidden)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=alpha)

        # Eligibility traces: one tensor per parameter
        self.traces = [
            torch.zeros_like(p, requires_grad=False)
            for p in self.q_net.parameters()
        ]

        self.name = f"TD(l={lambda_val})"

    # ------------------------------------------------------------------
    def reset_traces(self):
        """Zero all eligibility traces at the start of each episode."""
        for e in self.traces:
            e.zero_()

    # ------------------------------------------------------------------
    def select_action(self, obs: np.ndarray) -> int:
        """ε-greedy action selection."""
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
        """
        One-step TD(λ) update using accumulating eligibility traces.
        """
        s       = self._to_tensor(obs)
        s_next  = self._to_tensor(next_obs)

        # ---- Compute TD error δ ----------------------------------------
        q_values   = self.q_net(s)
        q_sa       = q_values[action]

        with torch.no_grad():
            q_next = self.q_net(s_next)
            v_next = q_next.max() if not done else torch.tensor(0.0)
        delta = reward + self.gamma * v_next.item() - q_sa.item()

        # ---- Compute gradient of Q(s,a) w.r.t. weights -----------------
        self.optimizer.zero_grad()
        q_sa.backward()

        # ---- Update elegibility traces and apply manual weight update ----
        with torch.no_grad():
            for param, trace in zip(self.q_net.parameters(), self.traces):
                if param.grad is not None:
                    # Accumulating trace: e ← γλe + ∇Q(s,a)
                    trace.mul_(self.gamma * self.lambda_val).add_(param.grad)
                    # Weight update: w ← w + α·δ·e
                    param.add_(self.optimizer.param_groups[0]["lr"] * delta * trace)

        # Zero gradients after manual update
        self.optimizer.zero_grad()

    # ------------------------------------------------------------------
    @staticmethod
    def _to_tensor(obs: np.ndarray) -> torch.Tensor:
        return torch.tensor(obs, dtype=torch.float32)

    def save(self, path: str):
        torch.save(self.q_net.state_dict(), path)

    def load(self, path: str):
        self.q_net.load_state_dict(torch.load(path))
