# Adaptive Portfolio Rebalancing Using TD(λ)
> Session II: Lab Activity — Eligibility Traces Exercise  
> SP Jain School of Global Management · Term 2 · RDMU

---

## Problem Statement

Design a Reinforcement Learning agent that learns to **rebalance a portfolio of assets** (stocks, ETFs) over time to **maximize long-term risk-adjusted returns**, using **TD(λ)** with eligibility traces.

---

## Project Structure

```
stock portfolio/
├── main.py                   ← Entry point (run this)
├── data_fetcher.py            ← Yahoo Finance market data downloader
├── portfolio_env.py           ← Custom Gym RL environment
├── td_lambda_agent.py         ← TD(λ) agent with eligibility traces + MLP
├── td0_agent.py               ← TD(0) baseline agent
├── monte_carlo_agent.py       ← Monte Carlo baseline agent
├── train.py                   ← Training loop & evaluation
├── visualize.py               ← Plot generation
├── why_eligibility_traces.md  ← Written answer to Assignment Q1
├── requirements.txt
├── results/                   ← Auto-created: PNGs + CSVs
└── models/                    ← Auto-created: saved model weights
```

---

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full pipeline
python main.py
```

### Optional CLI Arguments

```bash
python main.py --episodes 300 --tickers AAPL MSFT GOOGL SPY GLD --lambda_val 0.8
```

| Flag          | Default           | Description                        |
|---------------|-------------------|------------------------------------|
| `--episodes`  | 200               | Training episodes per agent        |
| `--tickers`   | AAPL MSFT GOOGL AMZN SPY | Yahoo Finance tickers      |
| `--start`     | 2020-01-01        | Historical data start date         |
| `--end`       | 2024-12-31        | Historical data end date           |
| `--lambda_val`| 0.7               | TD(λ) lambda value                 |

---

## Environment Design

| Component | Details |
|---|---|
| **State** | Current portfolio weights · daily log returns · 10-day rolling volatility · 10-day momentum |
| **Action** | Discrete: for each asset → decrease / hold / increase weight (±5%) |
| **Reward** | Daily portfolio return − risk penalty × volatility − transaction cost |
| **Episode** | 22 trading days (~1 month) |

---

## Agents Compared

| Agent | Method | Traces |
|---|---|---|
| **TD(λ)** | Multi-step TD with eligibility traces + neural net | ✅ Yes |
| **TD(0)** | One-step temporal difference + neural net | ❌ No |
| **Monte Carlo** | Full-episode return + neural net | ❌ No |

All agents use the **same 2-layer MLP** (128 hidden units) as the Q-function approximator, enabling a fair comparison.

---

## Output Files

After running `main.py`:

| File | Description |
|---|---|
| `results/learning_curves.png` | Episode reward curves for all three agents |
| `results/portfolio_value.png` | Portfolio growth ($1 invested) on test set |
| `results/comparison_summary.csv` | Final test rewards and portfolio values |
| `results/learning_curves.csv` | Per-episode reward data |
| `models/*.pt` | Saved PyTorch model weights |

---

## Why Eligibility Traces?

See **[why_eligibility_traces.md](why_eligibility_traces.md)** for the full written answer.

**Short answer:** Portfolio returns are delayed — a buying decision today may pay off days later. TD(0) misses this; Monte Carlo is too noisy. TD(λ) with eligibility traces bridges both: it propagates credit backwards through time, weighted by recency and frequency, leading to faster and more stable convergence in volatile markets.

---

## References

- Sutton & Barto (2018): *Reinforcement Learning: An Introduction*, Ch. 7 & 12
- [yfinance documentation](https://github.com/ranaroussi/yfinance)
- [Gymnasium](https://gymnasium.farama.org/)
- PyTorch documentation
