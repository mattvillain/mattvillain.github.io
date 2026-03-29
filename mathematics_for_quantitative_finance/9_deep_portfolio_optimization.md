# Deep Portfolio Optimisation

Markowitz solved portfolio optimisation for known means and covariances. In practice, we do not know them. Estimating expected returns from historical data is notoriously unreliable — the signal-to-noise ratio is so low that estimation errors dominate the optimisation, and the resulting portfolios are often worse than naive equal-weight allocations.

**Deep portfolio optimisation** bypasses the estimation step entirely. Instead of first estimating parameters and then optimising, a neural network learns the optimal portfolio allocation **directly from data**, end-to-end. The network takes raw features as input and outputs portfolio weights, trained to maximise a financial objective (Sharpe ratio, utility, or risk-adjusted return). This is a conceptual shift: from "estimate, then optimise" to "just optimise."

---

## End-to-End Portfolio Learning

:::{prf:definition} Deep Portfolio
:label: def-deep-portfolio

A **deep portfolio** is a neural network $w_\theta: \mathbb{R}^d \to \Delta^{N-1}$ that maps an input feature vector $x_t \in \mathbb{R}^d$ (observed at time $t$) to a vector of portfolio weights $w_t = w_\theta(x_t) \in \Delta^{N-1}$, where $\Delta^{N-1} = \{w \in \mathbb{R}^N_{\geq 0} : \sum_i w_i = 1\}$ is the probability simplex (for a long-only portfolio).

The features $x_t$ may include past returns, technical indicators, fundamental data, macroeconomic variables, or any other observable.
:::

The portfolio return over one period is $R_{p,t+1} = w_\theta(x_t)^\top R_{t+1}$, where $R_{t+1}$ is the vector of asset returns from $t$ to $t+1$.

### Objective Functions

The network is trained to optimise a **financial loss function** — not a prediction loss, but a portfolio performance metric.

:::{prf:definition} Portfolio Objective Functions
:label: def-portfolio-objectives

Common choices:

1. **Negative Sharpe ratio**:
$$\mathcal{L}(\theta) = -\frac{\hat{\mu}_{p,\theta}}{\hat{\sigma}_{p,\theta}} = -\frac{\frac{1}{T}\sum_t R_{p,t+1}^\theta}{\sqrt{\frac{1}{T}\sum_t (R_{p,t+1}^\theta - \hat{\mu}_{p,\theta})^2}}$$

2. **Expected utility**: $\mathcal{L}(\theta) = -\frac{1}{T}\sum_t u(R_{p,t+1}^\theta)$ for a concave utility function $u$ (e.g. $u(x) = \frac{x^{1-\gamma}}{1-\gamma}$ for power utility with risk aversion $\gamma$).

3. **Conditional Value at Risk (CVaR)**: $\mathcal{L}(\theta) = \text{CVaR}_\alpha(-R_{p,t+1}^\theta)$, the expected loss in the worst $\alpha$ fraction of outcomes.
:::

The Sharpe ratio is the most popular because it is scale-invariant and directly measures risk-adjusted performance. Training by gradient descent on the Sharpe ratio is differentiable (the formula is differentiable in the weights), so backpropagation works.

### Contrast with Two-Step Approaches

The classical approach to portfolio construction proceeds in two steps:
1. **Estimate** $\hat{\boldsymbol{\mu}}$ and $\hat{\Sigma}$ from historical data.
2. **Optimise** $w^* = \arg\min w^\top \hat{\Sigma} w$ s.t. $w^\top \hat{\boldsymbol{\mu}} \geq \mu_{\text{target}}$.

The problem is that estimation errors in $\hat{\boldsymbol{\mu}}$ propagate through the optimisation and get **amplified** — the optimizer "trusts" the estimates and concentrates the portfolio in assets with the highest estimated (but noisy) expected returns.

:::{prf:remark}
:label: rmk-end-to-end

End-to-end learning avoids this amplification. The network never explicitly estimates $\boldsymbol{\mu}$ or $\Sigma$; it learns whatever internal representations are useful for producing good portfolios. If expected returns are unpredictable (as the efficient market hypothesis suggests), the network can learn to focus on risk reduction and diversification — effectively rediscovering regularisation strategies like shrinkage, without being told to do so.

This is analogous to the shift from hand-crafted features to learned features in computer vision: instead of designing features (SIFT, HOG) and then classifying, end-to-end deep learning jointly learns features and classification. In portfolio theory, the "features" are $\boldsymbol{\mu}$ and $\Sigma$, and the "classification" is the portfolio choice.
:::

---

## Architecture Choices

### Output Layer: Enforcing Constraints

Portfolio weights must satisfy constraints: non-negativity (for long-only), budget ($\sum w_i = 1$), and possibly turnover or position limits.

:::{prf:definition} Constrained Portfolio Output
:label: def-portfolio-output

Common output-layer designs:

- **Long-only**: Apply $\operatorname{Softmax}$ to the network output: $w_i = e^{z_i} / \sum_j e^{z_j}$. This automatically satisfies $w_i \geq 0$ and $\sum w_i = 1$.
- **Long-short**: Apply a normalisation $w_i = z_i / \sum_j |z_j|$ with a leverage constraint, or use $w = \operatorname{Softmax}(z_+) - \operatorname{Softmax}(z_-)$ where $z = (z_+, z_-)$ splits the output into long and short components.
- **Turnover penalty**: Add $\lambda \sum_t \|w_{t+1} - w_t\|_1$ to the loss, penalising excessive rebalancing.
:::

The softmax output is particularly elegant: it maps any $\mathbb{R}^N$ vector into the simplex, is differentiable everywhere, and naturally handles the budget constraint. The temperature parameter of the softmax controls diversification: low temperature concentrates the portfolio, high temperature pushes toward equal weight.

### Temporal Models

Portfolio allocation is a **sequential** decision: today's allocation depends on past returns, current market conditions, and possibly previous positions (to manage turnover). This motivates temporal architectures.

- **LSTM / GRU**: Process a sequence of feature vectors $(x_1, \ldots, x_t)$ and output weights $w_t$. The hidden state carries information about market regime, trend, and volatility clustering.
- **Transformer**: Self-attention over the feature history. Can capture long-range dependencies and regime changes. The attention mechanism provides interpretable weights showing which past time steps most influence the current allocation.
- **Temporal Convolutional Networks (TCN)**: Dilated causal convolutions over the feature sequence. Fast to train, parallelisable, and effective for capturing multi-scale patterns.

---

## Reinforcement Learning for Portfolio Management

Portfolio rebalancing is naturally a **sequential decision problem**: at each time step, the agent observes the market state, takes an action (rebalance the portfolio), receives a reward (the portfolio return minus costs), and transitions to a new state. This is the textbook setup for reinforcement learning.

:::{prf:definition} Portfolio MDP
:label: def-portfolio-mdp

The portfolio management problem as an MDP:

- **State**: $s_t = (x_t, w_{t-1})$ — current features and previous portfolio weights.
- **Action**: $a_t = w_t \in \Delta^{N-1}$ — new portfolio weights.
- **Transition**: Market returns $R_{t+1}$ are drawn from the (unknown) environment dynamics.
- **Reward**: $r_t = w_t^\top R_{t+1} - c(w_t, w_{t-1})$ where $c$ is the transaction cost.
- **Objective**: Maximise $\mathbb{E}\!\left[\sum_{t=0}^{T-1} \gamma^t r_t\right]$ for discount factor $\gamma \leq 1$.
:::

Several RL algorithms have been applied:

- **Policy gradient (REINFORCE)**: Directly optimise the policy $\pi_\theta(a \mid s)$ by gradient ascent on the expected cumulative reward. Works well for continuous action spaces (portfolio weights).
- **Actor-Critic (A2C, PPO)**: Use a value network (critic) to reduce variance of the policy gradient. More stable training than pure policy gradient.
- **Deep Deterministic Policy Gradient (DDPG)**: For continuous action spaces, learns a deterministic policy $w_\theta(s_t)$ and a Q-function $Q_\psi(s, a)$ simultaneously.

:::{prf:remark}
:label: rmk-rl-challenges

RL for portfolio management faces several challenges that do not arise in standard RL benchmarks:

1. **Non-stationarity**: Financial markets are not stationary. A policy learned on one regime (e.g. low volatility bull market) may fail in another (e.g. crisis). Training on diverse historical periods or simulated regimes is essential.
2. **Reward sparsity**: Portfolio returns have very low signal-to-noise ratio. The RL agent must learn from a noisy reward signal.
3. **Transaction costs**: Realistic costs make the action space effectively continuous-with-friction, and the agent must learn to trade off rebalancing benefit against cost — the same challenge as in deep hedging.
4. **Sim-to-real gap**: Policies trained on simulated markets may not transfer to live trading. This is the financial analogue of the sim-to-real transfer problem in robotics.
:::

---

## Feature Importance and Interpretability

A deployed portfolio model must be **interpretable**: which features drive the allocation, and why? This is not only a regulatory requirement but a practical necessity — a model that allocates to an asset because of a spurious correlation should be caught before deployment.

### Shapley Values for Portfolio Explanations

**Shapley values** (from cooperative game theory) provide a principled attribution of each feature's contribution to the portfolio decision. Given a model output $w_\theta(x)$ and a feature vector $x = (x_1, \ldots, x_d)$, the Shapley value $\phi_j$ of feature $j$ measures its average marginal contribution:

$$\phi_j = \sum_{S \subseteq \{1,\ldots,d\} \setminus \{j\}} \frac{|S|!(d-|S|-1)!}{d!} \left[w_\theta(x_{S \cup \{j\}}) - w_\theta(x_S)\right]$$

where $x_S$ denotes the input with features outside $S$ marginalised out.

:::{prf:remark}
:label: rmk-shapley

Computing exact Shapley values is exponential in $d$ (the sum has $2^{d-1}$ terms). Efficient approximations exist — most notably **KernelSHAP** (Lundberg & Lee, 2017), which uses a weighted regression to estimate all $d$ Shapley values simultaneously.

Our own work on [Feature Importance for Time Series Data](https://arxiv.org/pdf/2210.02176) (ICAIF 2022) addresses a subtlety that arises in financial applications: when the model is a time-series model (e.g. an LSTM taking a window of past returns), treating each time step as an independent "player" is incorrect — the game-theoretic axioms of Shapley assume independent contributions, but time-series features are inherently ordered and correlated. We proposed methods for computing Shapley values that respect the temporal structure.

More recently, our [unified framework for Shapley estimation](https://arxiv.org/pdf/2506.05216) (NeurIPS 2025) provides provably efficient algorithms for computing Shapley values in high-dimensional settings — exactly the regime relevant for portfolio models with many assets and features.
:::

### Attention-Based Interpretability

For Transformer-based portfolio models, the **attention weights** provide a natural (if imperfect) interpretability mechanism. At each time step, the model assigns attention weights to past observations, revealing which historical periods the model considers most relevant for the current allocation decision.

---

## Connections and Outlook

Deep portfolio optimisation connects to nearly every other chapter in this book:

- **Portfolio theory** (ch. 4): Deep portfolios generalise Markowitz by learning the allocation function directly, rather than estimating parameters and then optimising.
- **Market completeness** (ch. 3): In a complete market, the optimal hedge is a linear readout of nonlinear basis functions — exactly a neural network. The optimal portfolio is a related but distinct object: it maximises utility rather than minimising replication error.
- **Deep hedging** (ch. 7): Hedging and portfolio optimisation are dual problems. Deep hedging minimises risk given a target payoff; deep portfolio optimisation maximises return given a risk budget.
- **Neural SDEs** (ch. 8): The asset dynamics model underlying the portfolio can itself be a neural SDE, creating a fully end-to-end system: learn the dynamics and the portfolio simultaneously.

The field is evolving rapidly. Open questions include:
- **Robustness**: How to ensure that learned portfolios are robust to distribution shift and adversarial market conditions.
- **Multi-agent**: What happens when multiple deep portfolio agents trade simultaneously? Game-theoretic equilibria in markets of neural networks.
- **Regulation**: How to satisfy regulatory constraints (risk limits, concentration limits, ESG) within the end-to-end learning framework.
