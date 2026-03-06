"""
train.py
========
Training loop for all three RL agents:
  1. TD(λ) — with eligibility traces
  2. TD(0) — one-step temporal difference (no traces)
  3. Monte Carlo — full-episode return updates

Saves per-episode reward curves and model weights.
Outputs a comparison CSV: results/comparison_summary.csv
"""

import os
import csv
import numpy as np

from portfolio_env   import PortfolioEnv
from td_lambda_agent import TDLambdaAgent
from td0_agent       import TD0Agent
from monte_carlo_agent import MonteCarloAgent


# -----------------------------------------------------------------------
# Hyper-parameters
# -----------------------------------------------------------------------
N_EPISODES   = 200        # training episodes per agent
GAMMA        = 0.99
ALPHA        = 1e-3
EPSILON      = 0.15
LAMBDA_VAL   = 0.7        # for TD(λ)
HIDDEN       = 128
RESULTS_DIR  = "results"
MODELS_DIR   = "models"


def run_episode(agent, env) -> tuple[float, list[float]]:
    """
    Run one full episode.
    Returns (total_reward, list_of_step_rewards).
    """
    obs, _ = env.reset()
    agent.reset_traces()
    done = False
    rewards = []

    while not done:
        action           = agent.select_action(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done             = terminated or truncated
        agent.update(obs, action, reward, next_obs, done)
        obs              = next_obs
        rewards.append(reward)

    return float(np.sum(rewards)), rewards


def train_agent(agent, env, n_episodes: int) -> list[float]:
    """Train an agent for n_episodes, return per-episode cumulative rewards."""
    episode_rewards = []
    for ep in range(1, n_episodes + 1):
        total, _ = run_episode(agent, env)
        episode_rewards.append(total)
        if ep % 20 == 0 or ep == 1:
            avg = np.mean(episode_rewards[-20:])
            print(f"  [{agent.name}] Episode {ep:>4}/{n_episodes} | "
                  f"Reward: {total:+.4f} | Avg-20: {avg:+.4f}")
    return episode_rewards


def evaluate_agent(agent, env, n_eval: int = 10) -> tuple[float, list[float]]:
    """
    Evaluate a trained agent (ε=0) over n_eval episodes.
    Returns (mean_reward, portfolio_value_trace).
    """
    orig_eps = agent.epsilon
    agent.epsilon = 0.0          # pure greedy

    total_rewards = []
    value_trace   = [1.0]        # start with $1 invested

    for _ in range(n_eval):
        obs, _ = env.reset()
        agent.reset_traces()
        done       = False
        ep_value   = 1.0
        ep_rewards = []

        while not done:
            action = agent.select_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_rewards.append(reward)
            ep_value *= (1.0 + info.get("portfolio_return", 0.0))

        total_rewards.append(np.sum(ep_rewards))
        value_trace.append(ep_value)

    agent.epsilon = orig_eps
    return float(np.mean(total_rewards)), value_trace


def main(n_episodes: int = N_EPISODES):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # ---- Load market data ------------------------------------------------
    from data_fetcher import load_and_prepare
    train_data, test_data = load_and_prepare()

    train_env = PortfolioEnv(train_data)
    test_env  = PortfolioEnv(test_data)

    obs_dim   = train_env.observation_space.shape[0]
    n_actions = train_env.action_space.n

    print(f"\nState dim: {obs_dim} | Actions: {n_actions}\n")

    # ---- Instantiate agents -----------------------------------------------
    td_lambda = TDLambdaAgent(obs_dim, n_actions,
                               lambda_val=LAMBDA_VAL, gamma=GAMMA,
                               alpha=ALPHA, epsilon=EPSILON, hidden=HIDDEN)
    td0       = TD0Agent(obs_dim, n_actions,
                          gamma=GAMMA, alpha=ALPHA,
                          epsilon=EPSILON, hidden=HIDDEN)
    mc        = MonteCarloAgent(obs_dim, n_actions,
                                 gamma=GAMMA, alpha=ALPHA,
                                 epsilon=EPSILON, hidden=HIDDEN)

    agents = [td_lambda, td0, mc]

    # ---- Train ------------------------------------------------------------
    all_rewards: dict[str, list[float]] = {}
    for agent in agents:
        print(f"\n{'='*55}")
        print(f"  Training: {agent.name}")
        print(f"{'='*55}")
        rewards = train_agent(agent, train_env, n_episodes)
        all_rewards[agent.name] = rewards
        agent.save(os.path.join(MODELS_DIR, f"{agent.name.replace('(','').replace(')','').replace('λ','l').replace('=','_')}.pt"))

    # ---- Save learning curves CSV ----------------------------------------
    lc_path = os.path.join(RESULTS_DIR, "learning_curves.csv")
    episodes = list(range(1, n_episodes + 1))
    with open(lc_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode"] + [a.name for a in agents])
        for i, ep in enumerate(episodes):
            writer.writerow([ep] + [all_rewards[a.name][i] for a in agents])
    print(f"\n[Results] Learning curves saved → {lc_path}")

    # ---- Evaluate on test set --------------------------------------------
    print("\n" + "="*55)
    print("  Evaluating on TEST SET")
    print("="*55)
    summary_rows = []
    value_traces: dict[str, list[float]] = {}

    for agent in agents:
        mean_r, val_trace = evaluate_agent(agent, test_env, n_eval=20)
        value_traces[agent.name] = val_trace
        summary_rows.append({"agent": agent.name, "mean_test_reward": mean_r,
                              "final_portfolio_value": val_trace[-1]})
        print(f"  {agent.name:<18} | Mean Reward: {mean_r:+.4f} "
              f"| Final Portfolio Value: {val_trace[-1]:.4f}")

    # ---- Save comparison summary CSV ------------------------------------
    summ_path = os.path.join(RESULTS_DIR, "comparison_summary.csv")
    with open(summ_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["agent", "mean_test_reward", "final_portfolio_value"])
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"[Results] Comparison summary saved → {summ_path}")

    # ---- Return data for visualize.py ------------------------------------
    return all_rewards, value_traces, [a.name for a in agents]


if __name__ == "__main__":
    all_rewards, value_traces, names = main()
    from visualize import plot_all
    plot_all(all_rewards, value_traces, names)
