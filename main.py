"""
main.py
=======
Entry point for the Adaptive Portfolio Rebalancing project.

Runs the full pipeline:
  1. Download market data (Yahoo Finance)
  2. Build RL environment
  3. Train TD(λ), TD(0), and Monte Carlo agents
  4. Evaluate on test set
  5. Generate comparison plots and CSVs

Usage:
  python main.py
  python main.py --episodes 300 --tickers AAPL MSFT GOOGL SPY

Optional CLI flags:
  --episodes   Number of training episodes per agent (default 200)
  --tickers    Space-separated list of Yahoo Finance tickers
  --start      Data start date (YYYY-MM-DD, default 2020-01-01)
  --end        Data end date   (YYYY-MM-DD, default 2024-12-31)
  --lambda_val TD(λ) lambda value (default 0.7)
"""

import argparse
import sys

# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Adaptive Portfolio Rebalancing Using TD(λ)"
    )
    parser.add_argument("--episodes",   type=int,   default=200,
                        help="Training episodes per agent")
    parser.add_argument("--tickers",    nargs="+",  default=["AAPL", "MSFT", "GOOGL", "AMZN", "SPY"],
                        help="Yahoo Finance tickers to include")
    parser.add_argument("--start",      type=str,   default="2020-01-01")
    parser.add_argument("--end",        type=str,   default="2024-12-31")
    parser.add_argument("--lambda_val", type=float, default=0.7)
    return parser.parse_args()


# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    print("\n" + "="*60)
    print("  Adaptive Portfolio Rebalancing Using TD(λ)")
    print("  Session II: Eligibility Traces — Lab Exercise")
    print("="*60)

    # ---- Step 1: Data --------------------------------------------------------
    print("\n[1/4] Fetching market data from Yahoo Finance...")
    from data_fetcher import load_and_prepare
    train_data, test_data = load_and_prepare(
        tickers=args.tickers,
        start=args.start,
        end=args.end,
    )

    # ---- Step 2 & 3: Train all agents ----------------------------------------
    print(f"\n[2/4] Training agents for {args.episodes} episodes each...")
    import train as tr
    # Temporarily override hyper-parameters from CLI
    tr.N_EPISODES  = args.episodes
    tr.LAMBDA_VAL  = args.lambda_val

    from portfolio_env   import PortfolioEnv
    from td_lambda_agent import TDLambdaAgent
    from td0_agent       import TD0Agent
    from monte_carlo_agent import MonteCarloAgent
    import os

    os.makedirs(tr.RESULTS_DIR, exist_ok=True)
    os.makedirs(tr.MODELS_DIR,  exist_ok=True)

    train_env = PortfolioEnv(train_data)
    test_env  = PortfolioEnv(test_data)

    obs_dim   = train_env.observation_space.shape[0]
    n_actions = train_env.action_space.n

    td_lambda = TDLambdaAgent(obs_dim, n_actions,
                               lambda_val=args.lambda_val,
                               gamma=tr.GAMMA, alpha=tr.ALPHA,
                               epsilon=tr.EPSILON, hidden=tr.HIDDEN)
    td0       = TD0Agent(obs_dim, n_actions,
                          gamma=tr.GAMMA, alpha=tr.ALPHA,
                          epsilon=tr.EPSILON, hidden=tr.HIDDEN)
    mc        = MonteCarloAgent(obs_dim, n_actions,
                                 gamma=tr.GAMMA, alpha=tr.ALPHA,
                                 epsilon=tr.EPSILON, hidden=tr.HIDDEN)

    agents      = [td_lambda, td0, mc]
    all_rewards = {}
    for agent in agents:
        print(f"\n  >> Training: {agent.name}")
        rewards = tr.train_agent(agent, train_env, args.episodes)
        all_rewards[agent.name] = rewards
        safe_name = agent.name.replace("(", "").replace(")", "").replace("λ", "l").replace("=", "_")
        agent.save(os.path.join(tr.MODELS_DIR, f"{safe_name}.pt"))

    # ---- Step 4: Evaluate on test set ----------------------------------------
    print("\n[3/4] Evaluating on test set...")
    import csv
    summary_rows: list[dict] = []
    value_traces: dict[str, list[float]] = {}

    for agent in agents:
        mean_r, val_trace = tr.evaluate_agent(agent, test_env, n_eval=20)
        value_traces[agent.name] = val_trace
        summary_rows.append({
            "agent": agent.name,
            "mean_test_reward": round(mean_r, 6),
            "final_portfolio_value": round(val_trace[-1], 6),
        })
        print(f"  {agent.name:<18} | Test Reward: {mean_r:+.4f} | "
              f"Portfolio Value: {val_trace[-1]:.4f}")

    summ_path = os.path.join(tr.RESULTS_DIR, "comparison_summary.csv")
    with open(summ_path, "w", newline="") as f:
        writer = csv.DictWriter(f,
            fieldnames=["agent", "mean_test_reward", "final_portfolio_value"])
        writer.writeheader()
        writer.writerows(summary_rows)

    # Save learning curves CSV
    import csv as _csv
    lc_path = os.path.join(tr.RESULTS_DIR, "learning_curves.csv")
    with open(lc_path, "w", newline="") as f:
        writer = _csv.writer(f)
        writer.writerow(["episode"] + [a.name for a in agents])
        for i in range(args.episodes):
            writer.writerow([i + 1] + [all_rewards[a.name][i] for a in agents])

    # ---- Step 5: Plots -------------------------------------------------------
    print("\n[4/4] Generating comparison plots...")
    from visualize import plot_all
    names = [a.name for a in agents]
    plot_all(all_rewards, value_traces, names)

    print("\n" + "="*60)
    print("  Done! Outputs saved to results/")
    print("    • results/learning_curves.png")
    print("    • results/portfolio_value.png")
    print("    • results/comparison_summary.csv")
    print("    • results/learning_curves.csv")
    print("  Model weights saved to models/")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
