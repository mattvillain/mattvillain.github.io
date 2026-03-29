# Deep Hedging

The Black-Scholes framework gives perfect hedges in a perfect market. But real markets have **transaction costs**, **discrete rebalancing**, **liquidity constraints**, and **model uncertainty**. Under any of these frictions, the Black-Scholes delta hedge is no longer optimal — and often not even well-defined, since the replicating portfolio may require infinite trading.

**Deep hedging** (Buehler, Gonon, Teichmann & Wood, 2019) replaces the analytical hedge with a **learned** one: a neural network that maps market states to hedge ratios, trained end-to-end to minimise a risk measure of the hedging error. It is, in my view, the cleanest application of deep learning to quantitative finance — and a natural synthesis of the market completeness and neural network themes of this book.

---

## The Hedging Problem

Consider a derivative with payoff $g(S_T)$ at maturity $T$. The hedger sells this derivative for price $p$ and must manage the resulting risk over $[0, T]$ by trading in the underlying $S$ at discrete times $0 = t_0 < t_1 < \cdots < t_n = T$.

At each time $t_k$, the hedger chooses a position $\delta_k$ in the underlying. The **hedging P\&L** at maturity is:

$$\text{P\&L} = p + \sum_{k=0}^{n-1} \delta_k (S_{t_{k+1}} - S_{t_k}) - g(S_T) - \text{costs}$$

where costs include transaction costs, funding costs, and any other frictions. The hedger wants to choose the strategy $(\delta_0, \ldots, \delta_{n-1})$ to make this P&L as "good" as possible — ideally zero (perfect replication), but in the presence of frictions, merely acceptable under some risk criterion.

---

## Deep Hedging Formulation

The key idea: parameterise the hedging strategy as a neural network and train it.

:::{prf:definition} Deep Hedging
:label: def-deep-hedging

The **deep hedging** problem is to find a neural network $\delta_\theta: \mathbb{R}^d \to \mathbb{R}$ mapping market features to hedge ratios, by solving:

$$\min_\theta \; \rho\!\left(-(p + H_T^\theta - g(S_T))\right)$$

where $\rho$ is a **convex risk measure** (e.g. CVaR, entropic risk), and

$$H_T^\theta = \sum_{k=0}^{n-1} \delta_\theta(I_{t_k}) \cdot (S_{t_{k+1}} - S_{t_k}) - c_k(\delta_\theta(I_{t_k}), \delta_\theta(I_{t_{k-1}}))$$

is the total hedging gain net of transaction costs $c_k$. The input features $I_{t_k}$ may include the current price $S_{t_k}$, time to maturity $T - t_k$, implied volatility, portfolio Greeks, or any other observable.
:::

The loss function is the risk measure applied to the negative hedging error. Minimising this loss trains the network to produce hedging strategies that are optimal under the specified risk criterion.

:::{prf:remark}
:label: rmk-bs-recovery

When $\rho = \operatorname{Var}$, costs are zero, and rebalancing is continuous, the optimal hedge recovers the **Black-Scholes delta**: $\delta^*(t, S) = \partial V / \partial S$. Deep hedging thus nests the classical theory as a special case, while generalising to arbitrary frictions and risk preferences.
:::

---

## Architecture and Training

### Network Architecture

At each time step $t_k$, the network receives the feature vector $I_{t_k}$ and outputs a hedge ratio $\delta_\theta(I_{t_k})$. The natural architecture is **recurrent**: a single network (shared weights) applied at each step, optionally with a hidden state that carries information across time steps.

Common choices:
- **Feedforward**: $\delta_\theta(I_{t_k})$ depends only on current features. Simple and effective for Markov underlyings.
- **LSTM / GRU**: Carries hidden state across time steps. Useful when the hedging decision depends on the path history (e.g. for path-dependent options or when features include moving averages).

The output is a scalar $\delta \in \mathbb{R}$ (the number of shares held). For multi-asset hedging, the output is a vector of positions in each hedging instrument.

### Training Procedure

:::{prf:definition} Training by Simulation
:label: def-dh-training

Deep hedging is trained by:

1. **Simulate** $M$ paths $\{S_{t_k}^{(i)}\}_{k,i}$ under a chosen model (e.g. GBM, Heston, or an empirical model).
2. **Forward pass**: For each path, compute $\delta_\theta(I_{t_k}^{(i)})$ at every step, accumulate the hedging P&L $H_T^{\theta,(i)}$.
3. **Compute loss**: Evaluate $\hat{\rho} = \frac{1}{M}\sum_i \ell(p + H_T^{\theta,(i)} - g(S_T^{(i)}))$ where $\ell$ is a sample-level loss consistent with the risk measure $\rho$.
4. **Backpropagate** through the entire sequence to update $\theta$.
:::

This is **backpropagation through time** applied to a financial simulation. The total gradient flows through every time step, through the transaction cost function, and through the risk measure.

---

## Transaction Costs and Incompleteness

The real power of deep hedging emerges in markets with **frictions**.

### Proportional Transaction Costs

With proportional costs (bid-ask spread), trading $\Delta\delta$ shares costs $c |\Delta\delta| S_t$ for some cost rate $c > 0$. The total cost is:

$$\text{Costs} = c \sum_{k=0}^{n-1} |\delta_{k+1} - \delta_k| \cdot S_{t_k}.$$

This introduces an $\ell_1$-type penalty on changes in the hedge ratio. The optimal strategy trades off replication accuracy against trading costs — it will **under-hedge** when the cost of adjusting exceeds the risk reduction.

:::{prf:remark}
:label: rmk-relu-hedging

Under proportional costs, the optimal hedging strategy exhibits **bang-bang** behaviour: the hedger either trades to the optimal hedge ratio or doesn't trade at all, depending on whether the current position is inside a **no-trade band** around the optimal frictionless delta. This threshold structure is **piecewise linear** in the position — reminiscent of a ReLU activation. Our own work on [ReLU network expressiveness](https://arxiv.org/pdf/2306.11827) (ECAI 2025) is relevant here: the space of piecewise-linear hedging strategies is exactly the space of ReLU network outputs, so a ReLU-based deep hedging model is architecturally matched to the structure of the optimal solution.
:::

### Incomplete Markets

In an **incomplete market** (e.g. stochastic volatility, jumps, or insufficient hedging instruments), perfect replication is impossible under any strategy. The hedging error has irreducible variance. Deep hedging handles this naturally: the network learns the **best achievable hedge** given the available instruments, with the risk measure trading off between expected cost and residual risk.

This connects directly to the market completeness discussion. The residual hedging error is the **approximation error** of the neural network: incomplete markets correspond to finite-width networks that cannot perfectly approximate the target payoff.

---

## The Indifference Price

Deep hedging also yields a natural notion of derivative pricing under frictions.

:::{prf:definition} Deep Hedging Indifference Price
:label: def-indifference-price

The **indifference price** $p^*$ of the derivative is the value of $p$ for which the hedger is indifferent between selling the derivative and hedging, versus doing nothing:

$$\rho\!\left(-(p^* + H_T^{\theta^*} - g(S_T))\right) = \rho(0)$$

where $\theta^*$ is the optimal network parameter. In practice, $p^*$ is found by optimising over both $\theta$ and $p$ jointly.
:::

Under zero transaction costs, the indifference price recovers the risk-neutral price $e^{-rT}\mathbb{E}^\mathbb{Q}[g(S_T)]$. Under frictions, it incorporates the cost of hedging into the price — a more realistic reflection of how derivatives are actually priced in practice.

---

## Practical Considerations

Several practical issues arise in deploying deep hedging:

**Model risk.** The network is trained on simulated paths from a specific model. If the real-world dynamics differ, the learned hedge may perform poorly. Mitigations include training on a **mixture of models** (GBM, Heston, jump-diffusion, etc.) to learn hedges that are robust across specifications, or training directly on **historical data** (with the caveat that historical paths are not drawn from the risk-neutral measure).

**Generalisation.** The network must hedge options with different strikes, maturities, and market conditions. Training on a distribution of contract parameters (rather than a single contract) produces a universal hedging network that generalises across the option surface.

**Interpretability.** What has the network learned? For vanilla options, one can plot $\delta_\theta(t, S)$ as a function of $S$ and compare it to the Black-Scholes delta. In practice, the learned hedge often resembles a "smoothed" or "bandwidth-adjusted" delta — the network has learned to reduce turnover by slightly underreacting to small price changes.
