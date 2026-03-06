"""
visualize.py
============
Plotting utilities for the portfolio rebalancing project.

Saves two PNG figures into the results/ directory:
  1. results/learning_curves.png  — per-episode reward for all three agents
  2. results/portfolio_value.png  — portfolio value growth on the test set
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")             # non-interactive backend (safe for all OS)
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


RESULTS_DIR = "results"
COLORS      = {
    "TD(l=0.7)":   "#2196F3",   # blue
    "TD(0)":        "#F44336",   # red
    "Monte Carlo":  "#4CAF50",   # green
}
LINESTYLES  = {
    "TD(l=0.7)":   "-",
    "TD(0)":        "--",
    "Monte Carlo":  "-.",
}


def smooth(values: list[float], window: int = 15) -> np.ndarray:
    """Simple moving average for smoother learning curves."""
    arr = np.array(values, dtype=float)
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="valid")


def plot_learning_curves(
    all_rewards: dict[str, list[float]],
    names: list[str],
    save_path: str | None = None,
):
    """Plot per-episode cumulative reward for each agent."""
    fig, ax = plt.subplots(figsize=(11, 5))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")

    for name in names:
        rewards = all_rewards[name]
        color   = COLORS.get(name, "#FFFFFF")
        ls      = LINESTYLES.get(name, "-")

        # Raw (faint)
        ax.plot(rewards, color=color, alpha=0.18, linewidth=0.8, linestyle=ls)
        # Smoothed
        sm = smooth(rewards, window=max(1, len(rewards) // 15))
        x  = np.arange(len(rewards) // 2, len(rewards) // 2 + len(sm))
        ax.plot(x, sm, color=color, linewidth=2.2, linestyle=ls, label=name)

    ax.set_xlabel("Episode", color="#adbac7", fontsize=11)
    ax.set_ylabel("Cumulative Reward", color="#adbac7", fontsize=11)
    ax.set_title("Learning Curves — TD(λ) vs TD(0) vs Monte Carlo",
                 color="#e6edf3", fontsize=13, fontweight="bold", pad=14)
    ax.tick_params(colors="#adbac7")
    ax.spines[:].set_color("#30363d")
    ax.grid(axis="y", color="#30363d", linewidth=0.6, linestyle="--")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))

    legend = ax.legend(framealpha=0.2, facecolor="#161b22",
                       edgecolor="#30363d", labelcolor="#e6edf3",
                       fontsize=10, loc="lower right")

    plt.tight_layout()
    path = save_path or os.path.join(RESULTS_DIR, "learning_curves.png")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[Visualize] Saved → {path}")


def plot_portfolio_value(
    value_traces: dict[str, list[float]],
    names: list[str],
    save_path: str | None = None,
):
    """Plot portfolio value growth ($1 invested) on the test set."""
    fig, ax = plt.subplots(figsize=(11, 5))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")

    for name in names:
        values = value_traces[name]
        color  = COLORS.get(name, "#FFFFFF")
        ls     = LINESTYLES.get(name, "-")
        ax.plot(values, color=color, linewidth=2.4, linestyle=ls, label=name)
        ax.fill_between(range(len(values)), 1.0, values,
                        color=color, alpha=0.07)

    ax.axhline(1.0, color="#888", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Evaluation Episode", color="#adbac7", fontsize=11)
    ax.set_ylabel("Portfolio Value (start = $1)", color="#adbac7", fontsize=11)
    ax.set_title("Portfolio Growth on Test Set",
                 color="#e6edf3", fontsize=13, fontweight="bold", pad=14)
    ax.tick_params(colors="#adbac7")
    ax.spines[:].set_color("#30363d")
    ax.grid(axis="y", color="#30363d", linewidth=0.6, linestyle="--")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.3f"))

    ax.legend(framealpha=0.2, facecolor="#161b22",
              edgecolor="#30363d", labelcolor="#e6edf3",
              fontsize=10, loc="upper left")

    plt.tight_layout()
    path = save_path or os.path.join(RESULTS_DIR, "portfolio_value.png")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[Visualize] Saved → {path}")


def plot_all(
    all_rewards: dict[str, list[float]],
    value_traces: dict[str, list[float]],
    names: list[str],
):
    plot_learning_curves(all_rewards, names)
    plot_portfolio_value(value_traces, names)


if __name__ == "__main__":
    # Quick smoke test with random data
    import random
    names   = ["TD(λ=0.7)", "TD(0)", "Monte Carlo"]
    rewards = {n: [random.gauss(0, 0.05) for _ in range(200)] for n in names}
    values  = {n: [1.0 + i * random.uniform(-0.01, 0.015) for i in range(21)]
               for n in names}
    plot_all(rewards, values, names)
