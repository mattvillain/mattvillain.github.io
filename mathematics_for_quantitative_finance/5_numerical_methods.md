# Numerical Methods

Closed-form solutions in mathematical finance are the exception, not the rule. The Black-Scholes formula for European options is a rare luxury. Most interesting pricing problems — path-dependent options, stochastic volatility models, American exercise features — require numerical computation. This chapter introduces the three pillars: **Monte Carlo simulation**, **SDE discretisation**, and **PDE finite differences**.

These methods are also the computational backbone of deep learning in finance. Training a neural SDE is discretising a stochastic differential equation (Euler-Maruyama). Pricing with a neural network is a form of Monte Carlo with learned dynamics. Understanding these classical methods is essential before we can discuss their modern, data-driven extensions.

---

## Monte Carlo Simulation

Monte Carlo methods estimate expectations by averaging over simulated samples. In finance, this means simulating asset paths under the risk-neutral measure $\mathbb{Q}$ and averaging discounted payoffs.

:::{prf:definition} Monte Carlo Estimator
:label: def-mc-estimator

Let $V_0 = e^{-rT} \mathbb{E}^\mathbb{Q}[g(S_T)]$ be the price of a European derivative with payoff $g$. The **Monte Carlo estimator** is:

$$\hat{V}_0^{(M)} = \frac{e^{-rT}}{M} \sum_{i=1}^M g(S_T^{(i)})$$

where $S_T^{(1)}, \ldots, S_T^{(M)}$ are independent samples of the terminal asset price under $\mathbb{Q}$.
:::

The beauty of Monte Carlo is its generality: it works for any payoff function $g$ and any model for which we can simulate paths. The cost is statistical: convergence is slow.

:::{prf:theorem} Monte Carlo Convergence
:label: thm-mc-convergence

Let $\sigma_g^2 = \operatorname{Var}^\mathbb{Q}(g(S_T)) < \infty$. Then:

1. **Consistency** (Strong Law): $\hat{V}_0^{(M)} \to V_0$ almost surely as $M \to \infty$.
2. **Rate** (Central Limit Theorem): $\sqrt{M}(\hat{V}_0^{(M)} - V_0) \xrightarrow{d} \mathcal{N}(0, e^{-2rT}\sigma_g^2)$.
3. The standard error of the estimator is $\operatorname{SE} = e^{-rT}\sigma_g / \sqrt{M}$, giving an $O(1/\sqrt{M})$ convergence rate.
:::

:::{prf:proof}
:class: dropdown
:enumerated: false

Since the $g(S_T^{(i)})$ are i.i.d. with mean $\mathbb{E}^\mathbb{Q}[g(S_T)]$ and finite variance $\sigma_g^2$, the Strong Law of Large Numbers gives $\frac{1}{M}\sum_i g(S_T^{(i)}) \to \mathbb{E}^\mathbb{Q}[g(S_T)]$ a.s. Multiplying by the constant $e^{-rT}$ gives consistency.

For the rate, the CLT gives $\sqrt{M}\left(\frac{1}{M}\sum_i g(S_T^{(i)}) - \mathbb{E}^\mathbb{Q}[g(S_T)]\right) \xrightarrow{d} \mathcal{N}(0, \sigma_g^2)$. Scaling by $e^{-rT}$ gives the claimed result. The standard error follows from $\operatorname{Var}(\hat{V}_0^{(M)}) = e^{-2rT}\sigma_g^2 / M$. $\square$
:::

The $O(1/\sqrt{M})$ rate is **dimension-independent** — this is Monte Carlo's great advantage over grid-based methods, which suffer exponentially from the curse of dimensionality. For a basket option on 100 underlyings, Monte Carlo works; a 100-dimensional PDE grid does not.

The downside is that $O(1/\sqrt{M})$ is slow: to halve the error, you must quadruple the number of samples. This motivates **variance reduction** techniques.

### Variance Reduction

The key insight is that we can estimate the same expectation with lower variance by being clever about how we sample.

:::{prf:definition} Antithetic Variates
:label: def-antithetic

If $S_T^{(i)}$ is generated using Gaussian increments $Z_1, \ldots, Z_n$, the **antithetic estimator** pairs each path with its mirror:

$$\hat{V}_0^{\text{AV}} = \frac{e^{-rT}}{M} \sum_{i=1}^{M/2} \frac{g(S_T^{(i)}) + g(\tilde{S}_T^{(i)})}{2}$$

where $\tilde{S}_T^{(i)}$ is generated using $-Z_1, \ldots, -Z_n$. If $g$ is monotone in the underlying (as for calls and puts), the negative correlation between paired samples reduces variance.
:::

:::{prf:definition} Control Variates
:label: def-control-variate

Given a random variable $C$ with known expectation $\mathbb{E}[C] = \mu_C$ that is correlated with $g(S_T)$, the **control variate estimator** is:

$$\hat{V}_0^{\text{CV}} = \frac{e^{-rT}}{M} \sum_{i=1}^M \left[g(S_T^{(i)}) - \beta(C^{(i)} - \mu_C)\right]$$

where $\beta = \operatorname{Cov}(g(S_T), C) / \operatorname{Var}(C)$ is the optimal control coefficient. The variance is reduced by a factor of $1 - \rho^2$, where $\rho$ is the correlation between $g(S_T)$ and $C$.
:::

A natural control variate for exotic option pricing is the corresponding vanilla option (whose price is known analytically). The correlation is typically high, yielding substantial variance reduction.

---

## Euler-Maruyama Discretisation

To simulate paths from an SDE $dX_t = \mu(t, X_t)\,dt + \sigma(t, X_t)\,dW_t$, we need to **discretise** time. The simplest scheme is Euler-Maruyama.

:::{prf:definition} Euler-Maruyama Scheme
:label: def-euler-maruyama

Given an SDE $dX_t = \mu(t, X_t)\,dt + \sigma(t, X_t)\,dW_t$ with initial condition $X_0 = x_0$, the **Euler-Maruyama discretisation** with step size $h = T/n$ produces the approximation:

$$X_{t_{k+1}}^h = X_{t_k}^h + \mu(t_k, X_{t_k}^h) \cdot h + \sigma(t_k, X_{t_k}^h) \cdot \sqrt{h} \, Z_{k+1}$$

where $t_k = kh$ and $Z_1, Z_2, \ldots \stackrel{\text{i.i.d.}}{\sim} \mathcal{N}(0,1)$.
:::

The scheme is intuitive: at each step, apply the deterministic drift $\mu \cdot h$ and the random diffusion $\sigma \cdot \sqrt{h} \cdot Z$. The $\sqrt{h}$ scaling reflects the scaling of Brownian increments: $W_{t+h} - W_t \sim \mathcal{N}(0, h)$.

:::{prf:theorem} Euler-Maruyama Convergence
:label: thm-em-convergence

Under standard Lipschitz and linear growth conditions on $\mu$ and $\sigma$:

1. **Strong convergence** of order $1/2$: $\mathbb{E}[|X_T - X_T^h|] = O(\sqrt{h})$.
2. **Weak convergence** of order $1$: $|\mathbb{E}[g(X_T)] - \mathbb{E}[g(X_T^h)]| = O(h)$ for smooth test functions $g$.
:::

Strong convergence measures path-by-path accuracy; weak convergence measures the accuracy of expectations (prices). For pricing, weak convergence is what matters, and Euler-Maruyama achieves first-order weak accuracy — halving the step size halves the pricing error.

:::{prf:remark}
:label: rmk-em-neural

The Euler-Maruyama scheme is directly relevant to deep learning. A **neural SDE** (covered in the chapter on [Neural SDEs](8_neural_sdes.md)) replaces $\mu$ and $\sigma$ with neural networks: $\mu_\theta(t, x)$ and $\sigma_\phi(t, x)$. Simulating a forward pass through the neural SDE is exactly an Euler-Maruyama discretisation. Training the network via backpropagation through these steps is **backpropagation through Euler-Maruyama** — the stochastic analogue of the adjoint method used in Neural ODEs (Chen et al., 2018).
:::

### Higher-Order Schemes

The **Milstein scheme** achieves strong order $1$ by including the first-order correction from Itô's lemma:

$$X_{t_{k+1}}^h = X_{t_k}^h + \mu \cdot h + \sigma \cdot \sqrt{h}\,Z_{k+1} + \frac{1}{2}\sigma \sigma' \left(h Z_{k+1}^2 - h\right)$$

where $\sigma' = \partial\sigma/\partial x$. For GBM (where $\sigma(x) = \sigma x$ and $\sigma' = \sigma$), the Milstein scheme gives exact log-normal samples — no discretisation error at all. This is why GBM simulation is straightforward.

---

## Finite Differences for PDEs

When the derivative price satisfies a PDE (like the Black-Scholes PDE), we can solve it numerically on a grid. This is the **finite difference method**.

:::{prf:definition} Finite Difference Grid
:label: def-fd-grid

Discretise the domain $[0, T] \times [0, S_{\max}]$ with time steps $\Delta t = T/M$ and spatial steps $\Delta S = S_{\max}/N$. Let $V_j^m \approx V(m\Delta t, j\Delta S)$ denote the approximate option price at grid point $(m, j)$.
:::

The Black-Scholes PDE $V_t + rSV_S + \frac{1}{2}\sigma^2 S^2 V_{SS} - rV = 0$ is approximated by replacing derivatives with finite differences:

- **Forward difference**: $V_t \approx (V_j^{m+1} - V_j^m) / \Delta t$
- **Central differences**: $V_S \approx (V_{j+1}^m - V_{j-1}^m) / (2\Delta S)$ and $V_{SS} \approx (V_{j+1}^m - 2V_j^m + V_{j-1}^m) / (\Delta S)^2$

### Explicit Scheme

The **explicit scheme** marches backward from the terminal condition $V_j^M = g(j\Delta S)$:

$$V_j^m = V_j^{m+1} + \Delta t \left[rj\Delta S \cdot \frac{V_{j+1}^{m+1} - V_{j-1}^{m+1}}{2\Delta S} + \frac{1}{2}\sigma^2 (j\Delta S)^2 \cdot \frac{V_{j+1}^{m+1} - 2V_j^{m+1} + V_{j-1}^{m+1}}{(\Delta S)^2} - rV_j^{m+1}\right].$$

This is simple to implement but subject to a **stability condition**: $\Delta t \leq (\Delta S)^2 / (\sigma^2 S_{\max}^2)$, which can force very small time steps.

### Implicit Scheme (Crank-Nicolson)

The **Crank-Nicolson scheme** averages the explicit and implicit discretisations, achieving second-order accuracy in both time and space while remaining unconditionally stable. At each time step, it requires solving a tridiagonal linear system — efficient via the Thomas algorithm.

:::{prf:remark}
:label: rmk-fd-deep

Finite difference methods for high-dimensional PDEs become intractable due to the curse of dimensionality: a grid in $d$ dimensions with $N$ points per dimension has $N^d$ total points. For basket options on many underlyings, this is prohibitive. The **deep BSDE method** (Han, Jentzen & E, 2018) addresses this by reformulating the PDE as a backward stochastic differential equation (BSDE) and approximating the solution with a neural network. The network learns the PDE solution at sampled points, avoiding the need for a full grid. This transforms a high-dimensional PDE problem into a stochastic optimisation problem — the natural habitat of deep learning.
:::
