"""
monte_carlo_agent.py
=====================
Monte Carlo agent — baseline comparison for TD(λ).

Waits until the episode ends, computes the full discounted return G_t
for each visited (state, action) pair, then performs a batch gradient
update. No bootstrapping is used.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from td_lambda_agent import QNetwork


class MonteCarloAgent:
    """
    Monte Carlo Q-learning agent with neural network function approximation.

    Collects full episode trajectories before computing and applying updates.

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

        # Trajectory buffer (cleared each episode)
        self._trajectory: list[tuple] = []

        self.name = "Monte Carlo"

    # ------------------------------------------------------------------
    def reset_traces(self):
        """Clear trajectory buffer at the start of each episode."""
        self._trajectory = []

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
        """
        For Monte Carlo: buffer the experience.
        The actual weight update is triggered only when done=True.
        """
        self._trajectory.append((obs.copy(), action, reward))

        if done:
            self._apply_mc_update()

    # ------------------------------------------------------------------
    def _apply_mc_update(self):
        """Compute discounted returns and apply batch gradient update."""
        if not self._trajectory:
            return

        # Compute returns G_t from end to start
        returns = []
        G = 0.0
        for _, _, reward in reversed(self._trajectory):
            G = reward + self.gamma * G
            returns.insert(0, G)

        # Normalize returns for stable training
        returns = np.array(returns, dtype=np.float32)
        if returns.std() > 1e-8:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Batch gradient update
        total_loss = torch.tensor(0.0)
        for (obs, action, _), G_t in zip(self._trajectory, returns):
            s       = self._to_tensor(obs)
            q_vals  = self.q_net(s)
            q_sa    = q_vals[action]
            target  = torch.tensor(G_t, dtype=torch.float32)
            total_loss = total_loss + self.loss_fn(q_sa, target)

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # Clear buffer
        self._trajectory = []

    # ------------------------------------------------------------------
    @staticmethod
    def _to_tensor(obs: np.ndarray) -> torch.Tensor:
        return torch.tensor(obs, dtype=torch.float32)

    def save(self, path: str):
        torch.save(self.q_net.state_dict(), path)

    def load(self, path: str):
        self.q_net.load_state_dict(torch.load(path))
