# Stochastic Calculus

Stochastic calculus is what happens when you try to integrate against a process that is continuous but nowhere differentiable. In classical calculus, the fundamental objects of integration are smooth paths. Financial markets are not smooth — they jitter continuously, tracing out paths that have no tangent anywhere. Attempting to use the classical Riemann–Stieltjes integral against such paths fails: the integral simply does not converge. We need to build a new theory.

This is not merely technical pedantry. The failure of classical integration reflects something deep about randomness: the quadratic variation of a Brownian path is non-trivial, and any calculus that ignores this will produce systematically wrong answers. The extra term in Itô's lemma — absent in classical calculus — is precisely the correction for this.

---

## Brownian Motion

The canonical continuous-time random process is **Brownian motion**, named after botanist Robert Brown who observed in 1827 that pollen particles suspended in water moved erratically, even in the absence of any external force. The mathematical model was developed by Einstein (1905) in his theory of diffusion, and the rigorous construction was given by Norbert Wiener (1923). It is sometimes called the **Wiener process** in his honour.

:::{prf:definition} Standard Brownian Motion
:label: def-brownian-motion

A stochastic process $(W_t)_{t \geq 0}$ on $(\Omega, \mathcal{F}, \mathbb{P})$ is a **standard Brownian motion** if:

1. $W_0 = 0$ almost surely.
2. **Independent increments**: for any $0 \leq t_0 < t_1 < \cdots < t_n$, the increments $W_{t_1} - W_{t_0}, \ldots, W_{t_n} - W_{t_{n-1}}$ are mutually independent.
3. **Stationary Gaussian increments**: $W_t - W_s \sim \mathcal{N}(0, t-s)$ for all $0 \leq s < t$.
4. **Continuous paths**: $t \mapsto W_t(\omega)$ is continuous almost surely.
:::

These axioms uniquely characterise the law of Brownian motion. A few remarkable consequences:

- **Nowhere differentiable**: Almost surely, $W_t$ is differentiable at no point. The paths are continuous but wildly rough.
- **Self-similar**: For any $c > 0$, $(W_{ct})_{t \geq 0} \stackrel{d}{=} (\sqrt{c}\, W_t)_{t \geq 0}$.
- **Martingale**: $W_t$ is a martingale with respect to its natural filtration.

The most important property for stochastic calculus is the **quadratic variation**.

:::{prf:definition} Quadratic Variation
:label: def-quadratic-variation

The **quadratic variation** of a process $(X_t)$ up to time $T$ is

$$[X]_T = \lim_{\|\Pi\| \to 0} \sum_{i=0}^{n-1} (X_{t_{i+1}} - X_{t_i})^2$$

where the limit is over partitions $\Pi: 0 = t_0 < t_1 < \cdots < t_n = T$ with mesh $\|\Pi\| \to 0$.
:::

:::{prf:theorem} Quadratic Variation of Brownian Motion
:label: thm-qv-bm

For standard Brownian motion, $[W]_T = T$ almost surely.
:::

:::{prf:proof}
:class: dropdown
:enumerated: false

Let $Q_\Pi = \sum_i (W_{t_{i+1}} - W_{t_i})^2$. Since increments are independent and $W_{t_{i+1}} - W_{t_i} \sim \mathcal{N}(0, \Delta t_i)$, we have:

- $\mathbb{E}[Q_\Pi] = \sum_i \Delta t_i = T$,
- $\operatorname{Var}(Q_\Pi) = \sum_i \operatorname{Var}((W_{t_{i+1}} - W_{t_i})^2) = \sum_i 2(\Delta t_i)^2 \leq 2 \|\Pi\| \cdot T \to 0$.

Thus $Q_\Pi \to T$ in $L^2$. Almost-sure convergence along dyadic partitions $\Pi_n$ (with mesh $T/2^n$) follows from the **Borel–Cantelli lemma**: by Chebyshev's inequality, $\mathbb{P}(|Q_{\Pi_n} - T| > \varepsilon) \leq \operatorname{Var}(Q_{\Pi_n})/\varepsilon^2 \leq 2T \cdot 2^{-n}/\varepsilon^2$, which is summable over $n$. $\square$
:::

The heuristic notation for this result is $dW_t \cdot dW_t = dt$, or simply $(dW_t)^2 = dt$. This is the fundamental rule of Itô calculus. Combined with $dt \cdot dt = 0$ and $dt \cdot dW_t = 0$, these are the **Itô multiplication rules**. They replace the classical $(dx)^2 = 0$.

---

## The Itô Integral

Since Brownian paths are not of bounded variation (indeed, they have infinite total variation almost surely), the classical Riemann–Stieltjes integral $\int_0^T H_t \, dW_t$ is not well-defined path-by-path. We need a construction that exploits the probabilistic structure.

:::{prf:definition} Itô Integral
:label: def-ito-integral

Let $(H_t)_{t \in [0,T]}$ be an adapted process with $\mathbb{E}\!\left[\int_0^T H_t^2 \, dt\right] < \infty$. The **Itô integral** $\int_0^T H_t \, dW_t$ is defined as the $L^2(\Omega)$ limit of left-endpoint Riemann sums:

$$\int_0^T H_t \, dW_t \coloneqq \lim_{\|\Pi\| \to 0} \sum_{i=0}^{n-1} H_{t_i} \left(W_{t_{i+1}} - W_{t_i}\right)$$

over partitions $\Pi$ of $[0,T]$.
:::

The left-endpoint convention is deliberate. It ensures that $H_{t_i}$ is $\mathcal{F}_{t_i}$-measurable and thus independent of the future increment $W_{t_{i+1}} - W_{t_i}$. This is the trading interpretation: you must choose your portfolio position $H_{t_i}$ before observing the next price move. The alternative midpoint convention (Stratonovich integral) has different properties and arises in the context of physical systems, but is less natural for finance.

The fundamental property of the Itô integral is the **Itô isometry**.

:::{prf:theorem} Itô Isometry
:label: thm-ito-isometry

For any adapted, square-integrable process $H$:

$$\mathbb{E}\!\left[\left(\int_0^T H_t \, dW_t\right)^2\right] = \mathbb{E}\!\left[\int_0^T H_t^2 \, dt\right].$$
:::

:::{prf:proof}
:class: dropdown
:enumerated: false

For a simple process $H_t = \sum_{i} h_i \mathbf{1}_{(t_i, t_{i+1}]}(t)$ with each $h_i \in \mathcal{F}_{t_i}$:

$$\left(\int_0^T H_t \, dW_t\right)^2 = \sum_{i,j} h_i h_j (W_{t_{i+1}} - W_{t_i})(W_{t_{j+1}} - W_{t_j}).$$

For $i \neq j$ (say $i < j$): taking expectations, $\mathbb{E}[h_i h_j (W_{t_{i+1}} - W_{t_i})(W_{t_{j+1}} - W_{t_j})] = \mathbb{E}[h_i h_j (W_{t_{i+1}} - W_{t_i})] \cdot \mathbb{E}[W_{t_{j+1}} - W_{t_j}] = 0$ since the increment $W_{t_{j+1}} - W_{t_j}$ is independent of $\mathcal{F}_{t_j}$ and has mean zero.

The diagonal terms give $\mathbb{E}[h_i^2 (W_{t_{i+1}} - W_{t_i})^2] = \mathbb{E}[h_i^2] \cdot \Delta t_i = \mathbb{E}[h_i^2 \Delta t_i]$, so the sum equals $\mathbb{E}[\int_0^T H_t^2 \, dt]$. The result extends to general $H$ by density of simple processes. $\square$
:::

The Itô isometry says that the map $H \mapsto \int H \, dW$ is an isometry from $L^2([0,T] \times \Omega)$ into $L^2(\Omega)$. This is the foundation of the $L^2$ theory: the stochastic integral is a well-defined, continuous linear operator.

An immediate corollary: the Itô integral $M_t = \int_0^t H_s \, dW_s$ is a **martingale** (provided $H$ is square-integrable). This is crucial — it means that stochastic integrals of adapted processes are always fair games.

---

## Itô's Lemma

Itô's lemma is the fundamental theorem of stochastic calculus — the stochastic analogue of the chain rule. Every calculation in quantitative finance ultimately invokes it.

:::{prf:theorem} Itô's Lemma
:label: thm-ito-lemma

Let $X_t$ be an Itô process:

$$dX_t = \mu_t \, dt + \sigma_t \, dW_t$$

and let $f \in C^{1,2}([0,T] \times \mathbb{R})$. Then $Y_t = f(t, X_t)$ is also an Itô process, and:

$$dY_t = \frac{\partial f}{\partial t}(t, X_t) \, dt + \frac{\partial f}{\partial x}(t, X_t) \, dX_t + \frac{1}{2} \frac{\partial^2 f}{\partial x^2}(t, X_t) \, \sigma_t^2 \, dt.$$

Substituting $dX_t$, this expands to:

$$dY_t = \left(\frac{\partial f}{\partial t} + \mu_t \frac{\partial f}{\partial x} + \frac{1}{2}\sigma_t^2 \frac{\partial^2 f}{\partial x^2}\right) dt + \sigma_t \frac{\partial f}{\partial x} \, dW_t.$$
:::

:::{prf:proof}
:class: dropdown
:enumerated: false

Apply a second-order Taylor expansion to $f(t + dt, X_{t+dt}) - f(t, X_t)$:

$$df = f_t \, dt + f_x \, dX_t + \frac{1}{2} f_{xx} \, (dX_t)^2 + \text{(higher order terms)}.$$

Compute $(dX_t)^2 = (\mu_t \, dt + \sigma_t \, dW_t)^2 = \mu_t^2 (dt)^2 + 2\mu_t \sigma_t \, dt \, dW_t + \sigma_t^2 (dW_t)^2$.

Applying the Itô multiplication table — $(dt)^2 = 0$, $dt \cdot dW_t = 0$, $(dW_t)^2 = dt$ — we get $(dX_t)^2 = \sigma_t^2 \, dt$. Substituting back yields the claimed expression. The higher-order terms are $o(dt)$ and vanish in the limit. $\square$
:::

The extra term $\frac{1}{2} \sigma_t^2 f_{xx} \, dt$ is the **Itô correction** (or **Itô drift**). It has no classical analogue and is the entire reason why stochastic calculus is its own subject. Ignoring it leads to systematic mispricing.

---

## Geometric Brownian Motion

The canonical model for equity prices in mathematical finance is **Geometric Brownian Motion (GBM)**. It captures the empirical observation that asset returns — not price changes — are approximately normally distributed, and that prices cannot go negative.

:::{prf:definition} Geometric Brownian Motion
:label: def-gbm

An asset price process $(S_t)_{t \geq 0}$ follows **Geometric Brownian Motion** if it satisfies the stochastic differential equation (SDE):

$$dS_t = \mu \, S_t \, dt + \sigma \, S_t \, dW_t, \quad S_0 > 0$$

where $\mu \in \mathbb{R}$ is the **drift** (expected continuously-compounded return) and $\sigma > 0$ is the **volatility**.
:::

The SDE says that the instantaneous relative return $dS_t / S_t$ has drift $\mu \, dt$ and noise $\sigma \, dW_t$. This is the continuous-time analogue of assuming log-normal returns.

:::{prf:theorem} Explicit Solution of GBM
:label: thm-gbm-solution

The unique strong solution to the GBM SDE with initial condition $S_0 > 0$ is:

$$S_t = S_0 \exp\!\left(\left(\mu - \frac{\sigma^2}{2}\right)t + \sigma W_t\right).$$
:::

:::{prf:proof}
:class: dropdown
:enumerated: false

Apply Itô's lemma to $f(t, x) = \log x$. The partial derivatives are $f_t = 0$, $f_x = 1/x$, $f_{xx} = -1/x^2$. With $dS_t = \mu S_t \, dt + \sigma S_t \, dW_t$:

$$d(\log S_t) = \frac{1}{S_t} \, dS_t + \frac{1}{2} \cdot \left(-\frac{1}{S_t^2}\right) \cdot \sigma^2 S_t^2 \, dt = \left(\mu - \frac{\sigma^2}{2}\right) dt + \sigma \, dW_t.$$

Since the right-hand side has constant coefficients, we can integrate directly:

$$\log S_t - \log S_0 = \left(\mu - \frac{\sigma^2}{2}\right)t + \sigma W_t.$$

Exponentiating: $S_t = S_0 \exp\!\bigl((\mu - \sigma^2/2)t + \sigma W_t\bigr)$. $\square$
:::

The **Itô correction** $-\sigma^2/2$ is worth pausing on. The expected value of $S_t$ is $\mathbb{E}[S_t] = S_0 e^{\mu t}$ (as expected from the SDE). But the **median** of $S_t$ is $S_0 \exp((\mu - \sigma^2/2)t)$, which grows more slowly. Higher volatility reduces the median growth rate — a mathematical statement of the arithmetic mean / geometric mean inequality for log-normal returns.

---

## The Black-Scholes Equation

Everything in stochastic calculus converges on the **Black-Scholes partial differential equation**, one of the most celebrated results in applied mathematics (and the basis for a Nobel Prize in Economics, 1997).

We first state the model assumptions explicitly.

:::{prf:definition} The Black-Scholes Model
:label: def-bs-model

The **Black-Scholes model** consists of the following assumptions:

1. The risky asset follows GBM: $dS_t = \mu S_t \, dt + \sigma S_t \, dW_t$ with constant $\mu$ and $\sigma > 0$.
2. There exists a **riskless bond** $B_t = e^{rt}$ earning a constant risk-free rate $r \geq 0$.
3. **Continuous trading** is permitted: portfolios can be rebalanced at every instant.
4. **No transaction costs**, no dividends, no taxes.
5. Borrowing and short-selling are unrestricted.
6. All securities are infinitely divisible.

Under these assumptions, the market is **complete**: by the Fundamental Theorem of Asset Pricing, the equivalent martingale measure $\mathbb{Q}$ is unique, and every contingent claim can be priced and perfectly hedged.
:::

Consider a **European call option** with payoff $(S_T - K)^+$ at maturity $T$. Under this model, the option's price $V(t, S)$ satisfies the following PDE.

:::{prf:theorem} Black-Scholes PDE
:label: thm-bs-pde

In the Black-Scholes model, the fair price $V(t, S)$ of any European derivative with terminal payoff $g(S_T)$ satisfies:

$$\frac{\partial V}{\partial t} + rS \frac{\partial V}{\partial S} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} - rV = 0, \quad t < T$$

with terminal condition $V(T, S) = g(S)$.
:::

:::{prf:proof}
:class: dropdown
:enumerated: false

Assume $V(t, S_t)$ is the price at time $t$. By Itô's lemma:

$$dV = \left(V_t + \mu S V_S + \frac{1}{2}\sigma^2 S^2 V_{SS}\right) dt + \sigma S V_S \, dW_t.$$

Construct the **delta-hedged portfolio** $\Pi_t = V - \Delta \cdot S_t$ where $\Delta = V_S$:

$$d\Pi_t = dV - \Delta \, dS_t = \left(V_t + \frac{1}{2}\sigma^2 S^2 V_{SS}\right) dt.$$

The portfolio is instantaneously riskless (no $dW_t$ term). By the **no-arbitrage principle**, any riskless portfolio must earn the risk-free rate $r$:

$$d\Pi_t = r \Pi_t \, dt = r(V - V_S \cdot S) \, dt.$$

Setting the two expressions equal:

$$V_t + \frac{1}{2}\sigma^2 S^2 V_{SS} = rV - rS V_S$$

which is the Black-Scholes PDE. $\square$
:::

The key step — constructing a riskless portfolio by delta-hedging — is conceptually elegant: the randomness in $V$ and $S$ is perfectly correlated (both driven by the same $W_t$), so a carefully chosen position in $S$ cancels the Brownian term in $dV$. What remains is deterministic, hence riskless.

The explicit solution for the European call is the **Black-Scholes formula**:

$$V(t, S) = S \, \Phi(d_+) - K e^{-r(T-t)} \Phi(d_-)$$

where

$$d_\pm = \frac{\log(S/K) + (r \pm \sigma^2/2)(T-t)}{\sigma\sqrt{T-t}}$$

and $\Phi$ is the standard normal CDF. This formula, derived by solving the PDE (or equivalently by computing $e^{-r(T-t)} \mathbb{E}^\mathbb{Q}[(S_T - K)^+]$ under the risk-neutral measure), is the benchmark of derivative pricing.

The connection between this formula and the structure of neural networks — via the Breeden-Litzenberger decomposition and the ReLU activation function — is the subject of the next chapter.
