"""
app.py  —  Streamlit Dashboard
================================
Adaptive Portfolio Rebalancing Using TD(λ)
Session II: Eligibility Traces — Lab Exercise

Sections:
  1. 📖  Why Eligibility Traces?  (written explanation)
  2. ⚙️  Configuration sidebar
  3. 📥  Data preview
  4. 🚀  Train all three agents (live progress)
  5. 📊  Learning curves (interactive Plotly)
  6. 💼  Portfolio value comparison (interactive Plotly)
  7. 🏆  Results summary table
"""

import os
import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Portfolio Rebalancing with TD(λ)",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.hero {
    background: linear-gradient(135deg, #1a1f35 0%, #0d1b2a 50%, #112240 100%);
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    border: 1px solid #1e3a5f;
}
.hero h1 { color: #64ffda; font-size: 2.2rem; font-weight: 700; margin: 0; }
.hero p  { color: #8892b0; font-size: 1.05rem; margin-top: 0.5rem; }
.hero .badge {
    display: inline-block;
    background: rgba(100,255,218,0.1);
    border: 1px solid #64ffda44;
    color: #64ffda;
    border-radius: 20px;
    padding: 3px 14px;
    font-size: 0.8rem;
    margin-top: 0.8rem;
    margin-right: 6px;
}

.section-card {
    background: #0d1b2a;
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 1.8rem;
    margin-bottom: 1.5rem;
}
.section-title {
    color: #ccd6f6;
    font-size: 1.2rem;
    font-weight: 600;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #1e3a5f;
}

.metric-card {
    background: linear-gradient(135deg, #112240, #0a192f);
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    text-align: center;
}
.metric-label { color: #8892b0; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1px; }
.metric-value { color: #64ffda; font-size: 1.8rem; font-weight: 700; }
.metric-sub   { color: #8892b0; font-size: 0.8rem; }

.why-block {
    background: #0a192f;
    border-left: 3px solid #64ffda;
    border-radius: 0 8px 8px 0;
    padding: 1rem 1.5rem;
    margin: 0.8rem 0;
    color: #a8b2d8;
    font-size: 0.95rem;
    line-height: 1.7;
}
.why-title { color: #64ffda; font-weight: 600; font-size: 1rem; margin-bottom: 0.3rem; }

.compare-table th { background: #112240 !important; color: #64ffda !important; }

stButton > button {
    background: linear-gradient(135deg, #64ffda, #00bfa5) !important;
    color: #0a192f !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 8px !important;
}
</style>
""", unsafe_allow_html=True)

# ── Hero Header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>📈 Adaptive Portfolio Rebalancing</h1>
    <p>Reinforcement Learning with Eligibility Traces · Session II Lab Exercise</p>
    <span class="badge">TD(λ)</span>
    <span class="badge">TD(0)</span>
    <span class="badge">Monte Carlo</span>
    <span class="badge">Yahoo Finance API</span>
    <span class="badge">Neural Network Q-Function</span>
</div>
""", unsafe_allow_html=True)


# ── Sidebar Configuration ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    tickers_input = st.text_input(
        "Tickers (comma-separated)",
        value="AAPL, MSFT, GOOGL, AMZN, SPY",
        help="Yahoo Finance ticker symbols"
    )
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.text_input("Start Date", value="2020-01-01")
    with col2:
        end_date = st.text_input("End Date", value="2024-12-31")

    st.markdown("---")
    n_episodes = st.slider("Training Episodes", min_value=50, max_value=500, value=150, step=50)
    lambda_val = st.slider("TD(λ) Lambda", min_value=0.0, max_value=1.0, value=0.7, step=0.05)
    gamma      = st.slider("Discount Factor (γ)", min_value=0.8, max_value=1.0, value=0.99, step=0.01)
    alpha      = st.select_slider("Learning Rate (α)", options=[1e-4, 5e-4, 1e-3, 5e-3, 1e-2], value=1e-3)
    epsilon    = st.slider("Exploration Rate (ε)", min_value=0.05, max_value=0.3, value=0.1, step=0.05)

    st.markdown("---")
    run_btn = st.button("🚀 Train All Agents", use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB LAYOUT
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["📖 Why Eligibility Traces?", "🚀 Train & Compare", "📋 About"])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Why Eligibility Traces?
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown("## 📖 Why Use Eligibility Traces for Portfolio Rebalancing?")

    st.markdown("""
    <div class="why-block">
        <div class="why-title">🕒 1. Delayed Rewards in Financial Markets</div>
        Portfolio decisions made <strong>today</strong> may produce rewards days or weeks later
        (e.g., a stock bought Monday rallies on Friday). TD(0) only propagates credit one step back —
        too myopic. Monte Carlo waits until the episode ends — too slow to adapt. <strong>TD(λ) propagates
        credit over a configurable horizon</strong>, perfectly matching the lag structure of market returns.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="why-block">
        <div class="why-title">🎯 2. Credit Assignment Across Correlated Decisions</div>
        A rebalancing decision (e.g., increasing allocation to tech stocks) may span multiple consecutive
        days before paying off. Eligibility traces ensure all the decisions that contributed to a profitable
        position receive <strong>proportional credit</strong>, not just the most recent one — solving the
        <em>temporal credit assignment problem</em>.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="why-block">
        <div class="why-title">📐 3. Frequency + Recency Weighting</div>
        In portfolio management, <strong>recency</strong> matters (older decisions are less responsible for
        current P&amp;L) and <strong>frequency</strong> matters (repeatedly choosing an action signals
        conviction). Eligibility traces encode both: recency via <code>γλ</code> decay, frequency via
        accumulating trace additions.
        <br><br>
        <code>e_t = γλ · e_{t-1} + ∇Q(s,a)</code>
        &nbsp;&nbsp;→&nbsp;&nbsp;
        <code>w ← w + α · δ_t · e_t</code>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="why-block">
        <div class="why-title">⚡ 4. Faster Convergence + Best Bias-Variance Trade-off</div>
        TD(λ) sits between TD(0) (high bias, low variance) and Monte Carlo (low bias, high variance).
        Tuning λ lets us find the optimal trade-off for the specific market data — yielding
        <strong>faster, more stable convergence</strong> in volatile environments.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📊 Method Comparison")

    df_compare = pd.DataFrame({
        "Method":           ["TD(0)", "TD(λ)", "Monte Carlo"],
        "Credit Lookback":  ["1 step only", "Multi-step (tunable λ)", "Full episode"],
        "Variance":         ["🟢 Low", "🟡 Medium", "🔴 High"],
        "Bias":             ["🔴 High", "🟡 Medium", "🟢 Low"],
        "Convergence":      ["🐢 Slow", "⚡ Fast", "🐢 Slow (noisy)"],
        "Uses Traces":      ["❌ No", "✅ Yes", "❌ No"],
    })
    st.dataframe(df_compare, width='stretch', hide_index=True)

    st.markdown("""
    > **Bottom line:** Eligibility traces give TD(λ) the ability to efficiently propagate rewards
    > backwards through a chain of portfolio decisions — the ideal algorithm for sequential financial
    > decision-making with delayed, noisy feedback.

    **Reference:** Sutton & Barto (2018) — *Reinforcement Learning: An Introduction*, Ch. 7 & 12.
    """)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Train & Compare
# ─────────────────────────────────────────────────────────────────────────────
with tab2:

    # ── Results from the pre-run (if they exist on disk) ──
    RESULTS_DIR = "results"

    def load_existing_results():
        """Load pre-computed results from disk if available."""
        lc_path   = os.path.join(RESULTS_DIR, "learning_curves.csv")
        summ_path = os.path.join(RESULTS_DIR, "comparison_summary.csv")
        if os.path.exists(lc_path) and os.path.exists(summ_path):
            lc   = pd.read_csv(lc_path)
            summ = pd.read_csv(summ_path)
            return lc, summ
        return None, None

    # ── Training trigger ──────────────────────────────────────────────────────
    if run_btn:
        st.markdown("### 🔄 Training in Progress...")

        # --- Data fetch ---
        data_status = st.status("📥 Downloading market data from Yahoo Finance...", expanded=True)
        with data_status:
            from data_fetcher import load_and_prepare
            train_data, test_data = load_and_prepare(
                tickers=tickers, start=start_date, end=end_date
            )
            st.write(f"✅ Train: **{len(train_data['dates'])} days** | Test: **{len(test_data['dates'])} days**")
            st.write(f"Assets: {', '.join(train_data['tickers'])}")
        data_status.update(label="✅ Market data downloaded!", state="complete")

        # --- Setup env & agents ---
        from portfolio_env   import PortfolioEnv
        from td_lambda_agent import TDLambdaAgent
        from td0_agent       import TD0Agent
        from monte_carlo_agent import MonteCarloAgent
        from train import run_episode, evaluate_agent

        train_env = PortfolioEnv(train_data)
        test_env  = PortfolioEnv(test_data)
        obs_dim   = train_env.observation_space.shape[0]
        n_actions = train_env.action_space.n

        agents = [
            TDLambdaAgent(obs_dim, n_actions, lambda_val=lambda_val,
                          gamma=gamma, alpha=alpha, epsilon=epsilon),
            TD0Agent(obs_dim,      n_actions, gamma=gamma, alpha=alpha, epsilon=epsilon),
            MonteCarloAgent(obs_dim, n_actions, gamma=gamma, alpha=alpha, epsilon=epsilon),
        ]
        COLORS = {"TD(l=0.7)": "#64ffda", "TD(0)": "#ff6b6b", "Monte Carlo": "#ffd166"}
        # update color key for current lambda
        COLORS[agents[0].name] = "#64ffda"

        all_rewards: dict[str, list[float]] = {a.name: [] for a in agents}

        # --- Per-agent training with progress bars ---
        for agent in agents:
            color = COLORS.get(agent.name, "#ffffff")
            with st.status(f"🤖 Training **{agent.name}**...", expanded=False) as ag_status:
                prog  = st.progress(0, text=f"{agent.name} — Episode 0/{n_episodes}")
                chart_placeholder = st.empty()
                window_rewards = []

                for ep in range(1, n_episodes + 1):
                    total, _ = run_episode(agent, train_env)
                    all_rewards[agent.name].append(total)
                    window_rewards.append(total)

                    pct = ep / n_episodes
                    prog.progress(pct, text=f"{agent.name} — Episode {ep}/{n_episodes} | Reward: {total:+.4f}")

                    # Update mini chart every 10 episodes
                    if ep % 10 == 0 or ep == n_episodes:
                        fig_mini = go.Figure()
                        fig_mini.add_trace(go.Scatter(
                            y=all_rewards[agent.name],
                            mode="lines",
                            line=dict(color=color, width=1.5),
                            name=agent.name,
                        ))
                        fig_mini.update_layout(
                            height=160, margin=dict(l=0,r=0,t=10,b=0),
                            plot_bgcolor="#0a192f", paper_bgcolor="#0a192f",
                            font_color="#a8b2d8", showlegend=False,
                            xaxis=dict(showgrid=False, color="#1e3a5f"),
                            yaxis=dict(showgrid=True, gridcolor="#1e3a5f", color="#1e3a5f"),
                        )
                        chart_placeholder.plotly_chart(fig_mini, use_container_width=True, key=f"mini_{agent.name}_{ep}")

                ag_status.update(label=f"✅ {agent.name} — Trained!", state="complete")

        # --- Save learning curves ---
        os.makedirs(RESULTS_DIR, exist_ok=True)
        lc_df = pd.DataFrame({"episode": range(1, n_episodes + 1)})
        for a in agents:
            lc_df[a.name] = all_rewards[a.name]
        lc_df.to_csv(os.path.join(RESULTS_DIR, "learning_curves.csv"), index=False)

        # --- Evaluate on test set ---
        eval_status = st.status("🧪 Evaluating on test set...", expanded=False)
        with eval_status:
            summary_rows = []
            value_traces: dict[str, list[float]] = {}
            for agent in agents:
                mean_r, val_trace = evaluate_agent(agent, test_env, n_eval=20)
                value_traces[agent.name] = val_trace
                summary_rows.append({
                    "Agent": agent.name,
                    "Mean Test Reward": round(mean_r, 6),
                    "Final Portfolio Value ($)": round(val_trace[-1], 4),
                })
                st.write(f"**{agent.name}** → Mean Reward: `{mean_r:+.4f}` | Portfolio Value: `${val_trace[-1]:.4f}`")
        eval_status.update(label="✅ Evaluation complete!", state="complete")

        summ_df = pd.DataFrame(summary_rows)
        summ_df.to_csv(os.path.join(RESULTS_DIR, "comparison_summary.csv"), index=False)

        # Store in session state
        st.session_state["lc_df"]       = lc_df
        st.session_state["summ_df"]     = summ_df
        st.session_state["value_traces"] = value_traces
        st.session_state["agents"]       = [a.name for a in agents]
        st.session_state["trained"]      = True
        st.success("🎉 Training complete! See results below.")

    # ── Display results (from session or from disk) ───────────────────────────
    AGENT_COLORS = {
        "TD(l=0.7)": "#64ffda",
        "TD(0)":     "#ff6b6b",
        "Monte Carlo": "#ffd166",
    }

    # Try session state first, then disk
    if "trained" in st.session_state and st.session_state["trained"]:
        lc_df       = st.session_state["lc_df"]
        summ_df     = st.session_state["summ_df"]
        value_traces = st.session_state["value_traces"]
        agent_names  = st.session_state["agents"]
        show_results = True
    else:
        lc_df, summ_raw = load_existing_results()
        show_results = lc_df is not None
        if show_results:
            agent_names = [c for c in lc_df.columns if c != "episode"]
            summ_df = summ_raw.rename(columns={
                "agent":                  "Agent",
                "mean_test_reward":       "Mean Test Reward",
                "final_portfolio_value":  "Final Portfolio Value ($)",
            })
            value_traces = None

    if show_results:
        st.markdown("---")

        # ── Metrics row ──
        if summ_df is not None and len(summ_df) > 0:
            st.markdown("### 🏆 Results Summary")
            cols = st.columns(len(summ_df))
            for i, (_, row) in enumerate(summ_df.iterrows()):
                agent_name = row.get("Agent", row.get("agent", ""))
                reward_val = row.get("Mean Test Reward", row.get("mean_test_reward", 0))
                port_val   = row.get("Final Portfolio Value ($)", row.get("final_portfolio_value", 0))
                color = AGENT_COLORS.get(agent_name, "#ffffff")
                with cols[i]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">{agent_name}</div>
                        <div class="metric-value" style="color:{color};">${port_val:.4f}</div>
                        <div class="metric-sub">Portfolio Value (start $1.00)</div>
                        <div class="metric-sub" style="margin-top:4px;">Reward: {float(reward_val):+.4f}</div>
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown("---")

        # ── Learning Curves ──
        st.markdown("### 📊 Learning Curves — Episode Rewards")

        def smooth(vals, w=15):
            arr = np.array(vals, dtype=float)
            if len(arr) < w:
                return arr
            return np.convolve(arr, np.ones(w) / w, mode="valid")

        fig_lc = go.Figure()
        for name in agent_names:
            color = AGENT_COLORS.get(name, "#ffffff")
            raw   = lc_df[name].tolist()
            # Raw (faint)
            fig_lc.add_trace(go.Scatter(
                y=raw, mode="lines", name=f"{name} (raw)",
                line=dict(color=color, width=0.7),
                opacity=0.25, showlegend=False,
            ))
            # Smoothed
            sm = smooth(raw).tolist()
            offset = len(raw) - len(sm)
            fig_lc.add_trace(go.Scatter(
                x=list(range(offset, len(raw))), y=sm,
                mode="lines", name=name,
                line=dict(color=color, width=2.5),
            ))

        fig_lc.update_layout(
            height=420,
            plot_bgcolor="#0a192f", paper_bgcolor="#0d1b2a",
            legend=dict(bgcolor="#0a192f", bordercolor="#1e3a5f", borderwidth=1,
                        font=dict(color="#a8b2d8")),
            font=dict(color="#a8b2d8"),
            xaxis=dict(title="Episode", showgrid=False, color="#8892b0",
                       zeroline=False),
            yaxis=dict(title="Cumulative Episode Reward", showgrid=True,
                       gridcolor="#1e3a5f", color="#8892b0", zeroline=False),
            margin=dict(l=10, r=10, t=20, b=10),
            hovermode="x unified",
        )
        st.plotly_chart(fig_lc, use_container_width=True)

        # ── Portfolio Value ──
        if value_traces:
            st.markdown("### 💼 Portfolio Value Growth (Test Set, $1 Invested)")
            fig_pv = go.Figure()
            for name in agent_names:
                color  = AGENT_COLORS.get(name, "#ffffff")
                values = value_traces[name]
                # Convert hex color to rgba for fill
            hex_c = color.lstrip("#")
            r_c, g_c, b_c = int(hex_c[:2], 16), int(hex_c[2:4], 16), int(hex_c[4:6], 16)
            fill_rgba = f"rgba({r_c},{g_c},{b_c},0.07)"
            fig_pv.add_trace(go.Scatter(
                    y=values, mode="lines+markers", name=name,
                    line=dict(color=color, width=2.5),
                    marker=dict(size=5, color=color),
                    fill="tozeroy", fillcolor=fill_rgba,
                ))
            fig_pv.add_hline(y=1.0, line_dash="dot", line_color="#8892b0",
                             annotation_text="Initial $1.00")
            fig_pv.update_layout(
                height=400,
                plot_bgcolor="#0a192f", paper_bgcolor="#0d1b2a",
                legend=dict(bgcolor="#0a192f", bordercolor="#1e3a5f", borderwidth=1,
                            font=dict(color="#a8b2d8")),
                font=dict(color="#a8b2d8"),
                xaxis=dict(title="Eval Episode", showgrid=False, color="#8892b0", zeroline=False),
                yaxis=dict(title="Portfolio Value ($)", showgrid=True,
                           gridcolor="#1e3a5f", color="#8892b0", zeroline=False, tickprefix="$"),
                margin=dict(l=10, r=10, t=20, b=10),
                hovermode="x unified",
            )
            st.plotly_chart(fig_pv, use_container_width=True)

        # ── Summary table ──
        st.markdown("### 📋 Comparison Table")
        st.dataframe(summ_df, width='stretch', hide_index=True)

        # Download buttons
        st.markdown("### 📥 Download Results")
        col_a, col_b = st.columns(2)
        with col_a:
            st.download_button(
                label="⬇️ Download Learning Curves CSV",
                data=lc_df.to_csv(index=False),
                file_name="learning_curves.csv",
                mime="text/csv",
            )
        with col_b:
            st.download_button(
                label="⬇️ Download Summary CSV",
                data=summ_df.to_csv(index=False),
                file_name="comparison_summary.csv",
                mime="text/csv",
            )
    else:
        st.info("👈 Configure your settings in the sidebar and click **🚀 Train All Agents** to begin.")
        st.markdown("""
        The dashboard will:
        1. **Download** real market data from Yahoo Finance
        2. **Train** TD(λ), TD(0), and Monte Carlo agents simultaneously
        3. **Evaluate** each agent on unseen test data
        4. **Plot** interactive learning curves and portfolio value charts
        5. **Export** results as downloadable CSV files
        """)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — About
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("## 📋 About This Project")
    col_l, col_r = st.columns([2, 1])
    with col_l:
        st.markdown("""
        ### Problem: Adaptive Portfolio Rebalancing Using TD(λ)
        **Objective:** Design a reinforcement learning agent that learns to rebalance a portfolio
        of assets (stocks, ETFs) over time to maximize long-term returns while minimizing risk.
        The agent uses **TD(λ)** to assign credit to past decisions based on both
        **frequency** and **recency** (eligibility traces).

        ### Environment
        | Component | Description |
        |---|---|
        | **State** | Portfolio weights · daily log returns · rolling volatility · momentum score |
        | **Action** | Discrete: increase / hold / decrease each asset's weight (±5%) |
        | **Reward** | Daily return − risk penalty × volatility − transaction cost |
        | **Episode** | 22 trading days (~1 calendar month) |

        ### Agents
        | Agent | Algorithm | Traces |
        |---|---|---|
        | **TD(l=λ)** | Q-learning with eligibility traces | ✅ Accumulating |
        | **TD(0)** | Standard one-step Q-learning | ❌ None |
        | **Monte Carlo** | Full-episode return updates | ❌ None |

        All agents use the **same 2-layer MLP** (128 hidden units) for fair comparison.
        """)
    with col_r:
        st.markdown("""
        ### Assignment Tasks
        - ✅ Explain why Eligibility Traces
        - ✅ Implement in Python
        - ✅ Integrate Yahoo Finance API
        - ✅ Function approximation (Neural Network)
        - ✅ Compare TD(λ) vs TD(0) vs Monte Carlo

        ### References
        - Sutton & Barto (2018)
          *RL: An Introduction* Ch. 7 & 12
        - [yfinance](https://github.com/ranaroussi/yfinance)
        - [Gymnasium](https://gymnasium.farama.org/)
        - [PyTorch](https://pytorch.org/)

        ### Course
        SP Jain School of Global Management  
        Term 2 · RDMU  
        Session II: Eligibility Traces
        """)
