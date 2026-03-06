# Why Use Eligibility Traces in Adaptive Portfolio Rebalancing?

## 1. What Are Eligibility Traces?

Eligibility traces are a **bridge between TD(0) (one-step temporal difference) and Monte Carlo methods**.  
They maintain a memory vector **e(s, a)** that records how recently and frequently each state–action pair was visited. When a reward is received, the TD error is **propagated back** through all recently visited states, weighted by how "eligible" they are.

The update rule for a weight vector **w** with eligibility trace **e** is:

```
δ_t  = r_{t+1} + γ·V(s_{t+1}) − V(s_t)      # TD error
e_t  = γ·λ·e_{t-1} + ∇V(s_t)               # trace update
w    ← w + α·δ_t·e_t                         # weight update
```

The parameter **λ ∈ [0, 1]** controls the decay:
- **λ = 0** → pure TD(0) (one-step look-ahead only)
- **λ = 1** → Monte Carlo (full episode return)

---

## 2. Why Eligibility Traces Are Ideal for Portfolio Rebalancing

### 2.1 Delayed Rewards in Financial Markets

Portfolio decisions made **today** often produce rewards days or weeks later (e.g., a stock bought Monday rises on Friday). TD(0) only propagates credit one step back — too myopic. Monte Carlo waits until the episode ends — too slow to adapt mid-month. **TD(λ) propagates credit over a configurable horizon**, perfectly matching the lag structure of market returns.

### 2.2 Credit Assignment Across Correlated Decisions

A rebalancing decision (e.g., increasing allocation to tech stocks) may span multiple consecutive days before paying off. Eligibility traces ensure all the decisions that contributed to a profitable position receive **proportional credit**, not just the most recent one. This is the **temporal credit assignment problem**, and eligibility traces solve it elegantly.

### 2.3 Frequency and Recency Weighting

In portfolio management:
- **Recency** matters: older allocation decisions are less responsible for current P&L.
- **Frequency** matters: repeatedly choosing an action signals conviction; those choices deserve more credit.

Eligibility traces naturally encode both:
- **Recency** via the exponential decay factor γλ.
- **Frequency** via accumulating traces (each visit adds ∇V to the trace vector).

### 2.4 Faster Convergence Than TD(0) and Monte Carlo

| Method | Credit Propagation | Variance | Bias | Convergence Speed |
|---|---|---|---|---|
| TD(0) | 1 step only | Low | High | Slow |
| Monte Carlo | Full episode | High | Low | Slow (noisy) |
| **TD(λ)** | **Multi-step (tunable)** | **Medium** | **Medium** | **Fast** |

TD(λ) finds the best trade-off between bias and variance, leading to **faster and more stable learning** in volatile market environments.

### 2.5 Compatible with Function Approximation

Portfolio state spaces are continuous and high-dimensional (prices, weights, volatility, sentiment). Eligibility traces work seamlessly with **neural network function approximation** via the gradient-based update rule above — making them practical for real-world trading agents.

---

## 3. Summary

> **Eligibility traces solve the temporal credit assignment problem that is inherent in sequential financial decision-making. They allow an RL agent to efficiently propagate reward signals backwards through a chain of portfolio decisions, weighting each by both recency and frequency. The result is faster learning, lower variance than Monte Carlo, and lower bias than TD(0) — a combination critical for adaptive portfolio rebalancing in noisy markets.**
