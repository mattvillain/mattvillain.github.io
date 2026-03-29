# American Options and Optimal Stopping

European options are exercised at maturity. American options can be exercised at any time up to maturity. This freedom to choose *when* to exercise creates a fundamentally different mathematical problem: the price of an American option is not a simple expectation, but the solution to an **optimal stopping problem**. The mathematics is richer, the computation harder, and the connections to reinforcement learning striking.

---

## The Optimal Stopping Problem

An American option with payoff function $g(S_t)$ (e.g. $g(S) = (K - S)^+$ for a put) gives the holder the right to exercise at any stopping time $\tau \leq T$. The holder wants to choose $\tau$ to maximise the expected discounted payoff.

:::{prf:definition} Stopping Time
:label: def-stopping-time

A random variable $\tau: \Omega \to [0, T]$ is a **stopping time** with respect to the filtration $\mathbb{F} = (\mathcal{F}_t)_{t \geq 0}$ if $\{\tau \leq t\} \in \mathcal{F}_t$ for every $t \geq 0$.
:::

The stopping time condition formalises the requirement that the exercise decision at time $t$ depends only on information available at time $t$ — no peeking into the future. You decide to exercise based on the current price, not tomorrow's.

:::{prf:definition} American Option Price
:label: def-american-price

The **fair price** of an American option with payoff $g$ at time $0$ is:

$$V_0^{\text{Am}} = \sup_{\tau \in \mathcal{T}_{[0,T]}} \mathbb{E}^\mathbb{Q}\!\left[e^{-r\tau} g(S_\tau)\right]$$

where $\mathcal{T}_{[0,T]}$ is the set of all stopping times taking values in $[0, T]$, and $\mathbb{Q}$ is the risk-neutral measure.
:::

This is a fundamentally different object from the European price $V_0^{\text{Eu}} = e^{-rT}\mathbb{E}^\mathbb{Q}[g(S_T)]$. The supremum over stopping times replaces the fixed-time expectation. Since the holder can always choose $\tau = T$ (exercise at maturity), we have $V_0^{\text{Am}} \geq V_0^{\text{Eu}}$ — the American option is always worth at least as much as its European counterpart. The **early exercise premium** is the difference.

---

## The Snell Envelope

The optimal stopping problem has a beautiful martingale characterisation via the **Snell envelope**.

:::{prf:definition} Snell Envelope
:label: def-snell-envelope

Given an adapted process $(Z_t)_{t \in [0,T]}$ (the "reward process"), its **Snell envelope** $(U_t)_{t \in [0,T]}$ is the smallest supermartingale dominating $Z$:

$$U_t = \operatorname*{ess\,sup}_{\tau \in \mathcal{T}_{[t,T]}} \mathbb{E}[Z_\tau \mid \mathcal{F}_t].$$
:::

In the American option setting, $Z_t = e^{-rt}g(S_t)$ is the discounted payoff from exercising at time $t$.

:::{prf:theorem} Optimal Stopping via the Snell Envelope
:label: thm-snell

Let $U_t$ be the Snell envelope of the discounted payoff process $Z_t = e^{-rt}g(S_t)$. Then:

1. $V_0^{\text{Am}} = U_0$ — the American price equals the initial value of the Snell envelope.
2. The **optimal exercise time** is $\tau^* = \inf\{t \geq 0 : U_t = Z_t\}$ — the first time the Snell envelope touches the payoff process.
3. $U$ satisfies the **dynamic programming principle**: $U_T = Z_T$ and for $t < T$, $U_t = \max(Z_t, \mathbb{E}[U_{t+1} \mid \mathcal{F}_t])$ (in discrete time).
:::

:::{prf:proof}
:class: dropdown
:enumerated: false

We prove this in discrete time $t \in \{0, 1, \ldots, n\}$ for clarity. Define $U_n = Z_n$ and backward:

$$U_k = \max\!\left(Z_k,\, \mathbb{E}[U_{k+1} \mid \mathcal{F}_k]\right).$$

**$U$ is a supermartingale:** $\mathbb{E}[U_{k+1} \mid \mathcal{F}_k] \leq \max(Z_k, \mathbb{E}[U_{k+1} \mid \mathcal{F}_k]) = U_k$.

**$U$ dominates $Z$:** $U_k \geq Z_k$ by construction.

**$U$ is the smallest such:** Let $\tilde{U}$ be any supermartingale dominating $Z$. At $n$: $\tilde{U}_n \geq Z_n = U_n$. Backward induction: if $\tilde{U}_{k+1} \geq U_{k+1}$, then $\mathbb{E}[\tilde{U}_{k+1} \mid \mathcal{F}_k] \geq \mathbb{E}[U_{k+1} \mid \mathcal{F}_k]$, and $\tilde{U}_k \geq Z_k$, so $\tilde{U}_k \geq \max(Z_k, \mathbb{E}[U_{k+1} \mid \mathcal{F}_k]) = U_k$.

**Optimality of $\tau^*$:** Define $\tau^* = \min\{k : U_k = Z_k\}$. The stopped process $(U_{k \wedge \tau^*})$ is a martingale (since $U$ is a supermartingale and is equal to $Z$ at $\tau^*$, the max in the recursion is never binding after $\tau^*$). Therefore $U_0 = \mathbb{E}[U_{\tau^*}] = \mathbb{E}[Z_{\tau^*}]$, and for any other $\tau$: $\mathbb{E}[Z_\tau] \leq \mathbb{E}[U_\tau] \leq U_0$ (since $U$ is a supermartingale). $\square$
:::

The dynamic programming recursion $U_k = \max(Z_k, \mathbb{E}[U_{k+1} \mid \mathcal{F}_k])$ has an intuitive interpretation: at each time step, the holder compares the **immediate exercise value** $Z_k$ with the **continuation value** $C_k = \mathbb{E}[U_{k+1} \mid \mathcal{F}_k]$. If exercising now is worth more, they exercise. Otherwise, they continue.

This "exercise or continue" decision is exactly the structure of a **Markov decision process** — a connection we will exploit below.

---

## The Longstaff-Schwartz Algorithm

The Snell envelope requires computing conditional expectations $\mathbb{E}[U_{k+1} \mid \mathcal{F}_k]$ at every time step — a high-dimensional regression problem. The **Longstaff-Schwartz algorithm** (2001) solves this via least-squares Monte Carlo.

:::{prf:definition} Longstaff-Schwartz Algorithm
:label: def-ls

The **Longstaff-Schwartz** (LSM) algorithm estimates American option prices by:

1. **Simulate** $M$ paths $\{S_{t_k}^{(i)}\}_{k=0}^n$, $i = 1, \ldots, M$, under $\mathbb{Q}$.
2. **Initialise** at maturity: $V_n^{(i)} = g(S_{t_n}^{(i)})$.
3. **Backward induction** for $k = n-1, \ldots, 1$:
   - Among paths where $g(S_{t_k}^{(i)}) > 0$ (in-the-money), regress the discounted future values $e^{-r\Delta t} V_{k+1}^{(i)}$ on a set of **basis functions** $\psi_1(S_{t_k}^{(i)}), \ldots, \psi_p(S_{t_k}^{(i)})$:
     $$\hat{C}_k(s) = \sum_{j=1}^p \hat{\beta}_j \psi_j(s) \approx \mathbb{E}^\mathbb{Q}[e^{-r\Delta t} V_{k+1} \mid S_{t_k} = s].$$
   - Exercise on path $i$ if $g(S_{t_k}^{(i)}) \geq \hat{C}_k(S_{t_k}^{(i)})$; otherwise continue.
   - Update: $V_k^{(i)} = g(S_{t_k}^{(i)})$ if exercised, $V_k^{(i)} = e^{-r\Delta t}V_{k+1}^{(i)}$ otherwise.
4. **Price**: $\hat{V}_0^{\text{Am}} = \frac{e^{-r\Delta t}}{M} \sum_i V_1^{(i)}$.
:::

The basis functions $\psi_j$ are typically polynomials (Laguerre, Hermite) or other simple functions of the state. The regression approximates the continuation value at each time step.

:::{prf:remark}
:label: rmk-ls-nn

The Longstaff-Schwartz regression is a **function approximation** problem: given samples $(S_{t_k}^{(i)}, e^{-r\Delta t}V_{k+1}^{(i)})$, learn the conditional expectation $\hat{C}_k(s)$. Using polynomials is the classical approach, but nothing prevents us from using **neural networks** as the function approximator instead. This is exactly what **deep optimal stopping** methods do (Becker, Cheridito & Jentzen, 2019): replace the least-squares regression at each time step with a neural network trained to predict the continuation value.

The advantage is expressiveness — neural networks can approximate the continuation value in high dimensions (e.g. for Bermudan swaptions on a full yield curve), where polynomial bases become intractable. The cost is training time and the usual challenges of neural network optimisation.
:::

---

## Connection to Reinforcement Learning

The optimal stopping problem has a natural interpretation as a **Markov Decision Process (MDP)**, and this connection opens the door to reinforcement learning methods.

:::{prf:definition} American Option as MDP
:label: def-american-mdp

The exercise decision for an American option defines an MDP:

- **State**: $s_t = (t, S_t)$ — the current time and asset price.
- **Actions**: $a \in \{\text{exercise}, \text{continue}\}$.
- **Transition**: Under the risk-neutral measure, $S_{t+\Delta t} \sim \mathbb{Q}(\cdot \mid S_t)$ (e.g. GBM).
- **Reward**: $r_t(s, a) = g(S_t) \cdot \mathbf{1}[a = \text{exercise}]$.
- **Objective**: Maximise $\mathbb{E}^\mathbb{Q}\!\left[\sum_{k=0}^n e^{-rt_k} r_{t_k}\right] = \mathbb{E}^\mathbb{Q}[e^{-r\tau}g(S_\tau)]$.

The **value function** $V(t, s) = \sup_\tau \mathbb{E}^\mathbb{Q}[e^{-r(\tau - t)}g(S_\tau) \mid S_t = s]$ satisfies the **Bellman equation**:

$$V(t, s) = \max\!\left(g(s),\; e^{-r\Delta t}\mathbb{E}^\mathbb{Q}[V(t + \Delta t, S_{t+\Delta t}) \mid S_t = s]\right).$$
:::

This is precisely the Snell envelope recursion! The Bellman equation and the dynamic programming principle are the same object, viewed from different traditions.

The reinforcement learning perspective suggests new algorithms:

- **Deep Q-learning**: Train a neural network $Q_\theta(t, s, a)$ to approximate the action-value function. The exercise policy is $\pi(t,s) = \arg\max_a Q_\theta(t,s,a)$.
- **Policy gradient**: Parameterise the exercise policy directly as $\pi_\theta(t, s) = \sigma(f_\theta(t,s))$ (a neural network with sigmoid output giving the probability of exercising), and optimise $\theta$ by gradient ascent on the expected payoff.

:::{prf:remark}
:label: rmk-rl-finance

The MDP formulation extends beyond American options. Portfolio rebalancing, execution timing (optimal trade scheduling), market making, and inventory management are all sequential decision problems. In each case, the financial agent observes a state (prices, inventory, order flow), takes an action (trade, hedge, quote), and receives a reward (P&L, risk-adjusted return). Reinforcement learning provides a principled framework for learning optimal policies in these settings — the subject of the chapter on [Deep Portfolio Optimisation](9_deep_portfolio_optimization.md).
:::

---

## Early Exercise: When Is It Optimal?

A natural question: when should one exercise an American option early?

:::{prf:theorem} Early Exercise of American Calls
:label: thm-no-early-exercise

For an American call option on a non-dividend-paying stock, early exercise is **never optimal**: $V^{\text{Am}}_{\text{call}} = V^{\text{Eu}}_{\text{call}}$.
:::

:::{prf:proof}
:class: dropdown
:enumerated: false

For any $t < T$, the value of holding the call is at least $\mathbb{E}^\mathbb{Q}[e^{-r(T-t)}(S_T - K)^+ \mid \mathcal{F}_t]$. By Jensen's inequality applied to the convex function $x \mapsto (x - K)^+$ and the fact that $e^{-r(T-t)}\mathbb{E}^\mathbb{Q}[S_T \mid \mathcal{F}_t] = S_t$ (the discounted price is a $\mathbb{Q}$-martingale):

$$\mathbb{E}^\mathbb{Q}[e^{-r(T-t)}(S_T - K)^+ \mid \mathcal{F}_t] \geq (S_t - Ke^{-r(T-t)})^+ > (S_t - K)^+$$

whenever $r > 0$ and $T > t$. So the continuation value strictly exceeds the exercise value. $\square$
:::

American puts, by contrast, can be optimally exercised early — deep in-the-money, the opportunity cost of locking in the exercise value outweighs the optionality. The **free boundary** separating the exercise region from the continuation region is itself a function of time and must be determined as part of the solution.
