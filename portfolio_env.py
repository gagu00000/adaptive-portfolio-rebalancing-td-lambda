"""
portfolio_env.py
================
Custom Gymnasium-compatible RL environment for adaptive portfolio rebalancing.

State  (s): [current_weights | daily_returns | rolling_volatility | momentum]
            Shape: (4 * n_assets,)

Action (a): Discrete — for each asset: 0 = decrease, 1 = hold, 2 = increase
            Encoded as a single integer in [0, 3^n_assets)

Reward (r): Risk-adjusted daily portfolio return
            r = portfolio_return - risk_penalty * portfolio_volatility

Episode:    EPISODE_LENGTH trading days (default = 22, ~1 month)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


EPISODE_LENGTH = 22      # ~1 trading month
TRADE_COST     = 0.001   # 0.1% transaction cost per rebalance
WEIGHT_DELTA   = 0.05    # weight change per action step (±5%)
RISK_PENALTY   = 0.5     # coefficient on volatility in reward


class PortfolioEnv(gym.Env):
    """
    Adaptive Portfolio Rebalancing Environment.

    Parameters
    ----------
    data : dict
        Output from data_fetcher.load_and_prepare() — contains 'returns',
        'volatility', 'momentum', 'tickers'.
    episode_length : int
        Number of trading days per episode.
    """

    metadata = {"render_modes": []}

    def __init__(self, data: dict, episode_length: int = EPISODE_LENGTH):
        super().__init__()

        self.returns    = data["returns"].to_numpy(dtype=np.float32)
        self.volatility = data["volatility"].to_numpy(dtype=np.float32)
        self.momentum   = data["momentum"].to_numpy(dtype=np.float32)
        self.tickers    = data["tickers"]
        self.n_assets   = len(self.tickers)
        self.episode_length = episode_length

        # Maximum possible start index
        self.max_start = len(self.returns) - episode_length - 1

        # Action space: discrete — maps integer → weight-change vector
        self._build_action_map()
        self.action_space = spaces.Discrete(self.n_actions)

        # Observation space: weights + returns + volatility + momentum
        obs_dim = 4 * self.n_assets
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self._reset_state()

    # ------------------------------------------------------------------
    # Action mapping: each asset independently gets {-1, 0, +1} weight delta
    # Total actions = 3^n_assets
    # ------------------------------------------------------------------
    def _build_action_map(self):
        n = self.n_assets
        self.n_actions = 3 ** n
        self.action_map = []
        for i in range(self.n_actions):
            deltas = []
            code = i
            for _ in range(n):
                deltas.append(code % 3 - 1)   # maps {0,1,2} → {-1,0,+1}
                code //= 3
            self.action_map.append(np.array(deltas, dtype=np.float32))

    # ------------------------------------------------------------------
    def _reset_state(self):
        self.weights = np.ones(self.n_assets, dtype=np.float32) / self.n_assets
        self.t       = 0
        self.start   = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        # Random starting point in the dataset
        self.start = self.np_random.integers(0, self.max_start)
        self.t     = 0
        return self._obs(), {}

    def _obs(self) -> np.ndarray:
        idx = self.start + self.t
        return np.concatenate([
            self.weights,
            self.returns[idx],
            self.volatility[idx],
            self.momentum[idx],
        ]).astype(np.float32)

    def step(self, action: int):
        delta = self.action_map[action]

        # Apply weight deltas and project onto simplex
        old_weights    = self.weights.copy()
        new_weights    = np.clip(self.weights + delta * WEIGHT_DELTA, 0, 1)
        new_weights   /= new_weights.sum() + 1e-8

        # Transaction cost (L1 distance of weight change)
        turnover       = np.sum(np.abs(new_weights - old_weights))
        cost           = TRADE_COST * turnover

        self.weights   = new_weights
        idx            = self.start + self.t
        daily_returns  = self.returns[idx]
        daily_vol      = self.volatility[idx].mean()

        # Portfolio return
        port_return    = float(np.dot(self.weights, daily_returns))
        reward         = port_return - RISK_PENALTY * daily_vol - cost

        self.t        += 1
        terminated     = self.t >= self.episode_length
        truncated      = False

        return self._obs(), reward, terminated, truncated, {
            "portfolio_return": port_return,
            "weights": self.weights.copy(),
        }

    def render(self):
        print(f"[t={self.t}] weights={np.round(self.weights, 3)}")
