# Probability for Finance

There are millions of courses on probability theory. Therefore, I will be slightly exotic here — and if you do not like what you see, feel free to return to one of the many great standard guides. What I want to do is build up the theory in the way that makes the financial applications feel inevitable, and along the way point to the structures that connect back to machine learning.

Probabilities are **models of uncertainty**. The world is unbearably complex; often, instead of attempting to compute extremely complex phenomena — the exact trajectory of a coin toss, the precise dynamics of an equity order book — we make simplifying yet useful assumptions. This is *epistemic* uncertainty: uncertainty that arises from the limits of our knowledge. There is also *aleatoric* uncertainty, which is intrinsic to the phenomenon itself, as occurs at the quantum scale. The word aleatoric comes from *alea*, meaning dice in Latin — as in *Alea iacta est*, Caesar's famous words before crossing the Rubicon.

In finance, probability models the randomness of asset prices, interest rates, and default events. The machinery required is non-trivial: it involves measure theory, filtrations, martingales, and changes of measure. We build it up here carefully.

---

## Probability Spaces

The foundational object is the probability space. It has three components: a universe of outcomes, a notion of which events are observable, and a function that assigns likelihoods.

:::{prf:definition} Probability Space
:label: def-probability-space

A **probability space** is a triple $(\Omega, \mathcal{F}, \mathbb{P})$ composed of:

- $\Omega$: the **sample space**, the set of all elementary outcomes. It can be finite, countably infinite, or uncountable.
- $\mathcal{F} \subseteq 2^\Omega$: a **$\sigma$-algebra** on $\Omega$, the collection of *measurable events*. It must satisfy:
    - $\Omega, \emptyset \in \mathcal{F}$,
    - *Closure under complementation*: $A \in \mathcal{F} \implies A^c \in \mathcal{F}$,
    - *Closure under countable unions*: $\{A_i\}_{i \geq 1} \subset \mathcal{F} \implies \bigcup_{i=1}^\infty A_i \in \mathcal{F}$.
- $\mathbb{P}: \mathcal{F} \to [0,1]$: a **probability measure** satisfying:
    - $\mathbb{P}(\Omega) = 1$,
    - *Countable additivity*: if $\{A_i\} \subset \mathcal{F}$ are pairwise disjoint, then $\mathbb{P}\!\left(\bigcup_{i=1}^\infty A_i\right) = \sum_{i=1}^\infty \mathbb{P}(A_i)$.
:::

The $\sigma$-algebra $\mathcal{F}$ encodes what we can *ask* about the world — which events are distinguishable and measurable. This is not a mere technicality: in financial models, different $\sigma$-algebras represent different information states, and what you know determines what you can hedge. Two investors observing different information live, mathematically, in different $\sigma$-algebras.

A standard example: take $\Omega = \{H, T\}^n$ (all sequences of $n$ coin flips). The power set $2^\Omega$ is the largest possible $\sigma$-algebra — it distinguishes every outcome. A smaller $\sigma$-algebra might only distinguish the total number of heads, collapsing sequences with the same count into equivalence classes.

---

## Random Variables

:::{prf:definition} Random Variable
:label: def-random-variable

Given a probability space $(\Omega, \mathcal{F}, \mathbb{P})$, a **random variable** is a measurable function $X: \Omega \to \mathbb{R}$, meaning $X^{-1}(B) \coloneqq \{\omega \in \Omega : X(\omega) \in B\} \in \mathcal{F}$ for every Borel set $B \subseteq \mathbb{R}$.

The **distribution** (or **law**) of $X$ is the pushforward measure $\mathbb{P}_X \coloneqq \mathbb{P} \circ X^{-1}$ on $(\mathbb{R}, \mathcal{B}(\mathbb{R}))$.
:::

Random variables model observable quantities: an asset price $S_T$ at maturity $T$, a portfolio return over one month, a binary default indicator. Their distributions encode all probabilistic information about those quantities.

:::{prf:definition} Expectation
:label: def-expectation

The **expectation** of a random variable $X$ is

$$\mathbb{E}[X] = \int_\Omega X(\omega) \, d\mathbb{P}(\omega) = \int_\mathbb{R} x \, d\mathbb{P}_X(x)$$

whenever the Lebesgue integral is well-defined (i.e. $\mathbb{E}[|X|] < \infty$). The **variance** is $\operatorname{Var}(X) = \mathbb{E}[(X - \mathbb{E}[X])^2]$.
:::

---

## Filtrations and Information

In dynamic settings — financial markets evolving in continuous time — we need to model the **flow of information** over time. A particle of information arriving at time $t$ should become part of our knowledge from that point forward, but not before. A filtration does exactly this.

:::{prf:definition} Filtration
:label: def-filtration

A **filtration** on $(\Omega, \mathcal{F}, \mathbb{P})$ is a family $\mathbb{F} = (\mathcal{F}_t)_{t \geq 0}$ of sub-$\sigma$-algebras of $\mathcal{F}$ satisfying the **monotonicity** condition: $\mathcal{F}_s \subseteq \mathcal{F}_t$ for all $0 \leq s \leq t$.

A stochastic process $(X_t)_{t \geq 0}$ is **adapted** to $\mathbb{F}$ if $X_t$ is $\mathcal{F}_t$-measurable for each $t \geq 0$.
:::

Intuitively, $\mathcal{F}_t$ contains all information available up to time $t$, and the monotonicity says that information is never forgotten. Adaptedness means the value of $X_t$ is determined solely by information available at time $t$ — you cannot see the future.

A stronger requirement, **predictability**, asks that $X_t$ is $\mathcal{F}_{t-}$-measurable (where $\mathcal{F}_{t-} = \sigma(\bigcup_{s < t} \mathcal{F}_s)$), meaning the value at time $t$ is determined by information strictly before $t$. Trading strategies must be predictable: you cannot react instantaneously to a price movement to generate arbitrage.

The canonical filtration is the **natural filtration** of a process $(X_t)$: $\mathcal{F}_t^X = \sigma(X_s : s \leq t)$, the smallest $\sigma$-algebra making all $X_s$ for $s \leq t$ measurable. It represents observing the process and nothing else.

---

## Conditional Expectation

The conditional expectation is one of the most important constructions in probability. In finance, it is precisely how we compute the **fair price** of a derivative given current information.

:::{prf:definition} Conditional Expectation
:label: def-conditional-expectation

Let $X \in L^1(\Omega, \mathcal{F}, \mathbb{P})$ and let $\mathcal{G} \subseteq \mathcal{F}$ be a sub-$\sigma$-algebra. The **conditional expectation** $\mathbb{E}[X \mid \mathcal{G}]$ is the (a.s. unique) $\mathcal{G}$-measurable random variable $Z$ satisfying the **partial averaging property**:

$$\int_A Z \, d\mathbb{P} = \int_A X \, d\mathbb{P} \quad \text{for all } A \in \mathcal{G}.$$
:::

Existence and uniqueness follow from the Radon-Nikodym theorem. The conditional expectation can be understood as the **$L^2$-projection** of $X$ onto the closed subspace of $\mathcal{G}$-measurable square-integrable random variables: it is the best prediction of $X$ given information $\mathcal{G}$ in the mean-squared-error sense.

The tower property is the key computational rule.

:::{prf:theorem} Tower Property (Law of Iterated Expectations)
:label: thm-tower

For $\mathcal{G}_1 \subseteq \mathcal{G}_2 \subseteq \mathcal{F}$ and $X \in L^1$:

$$\mathbb{E}\bigl[\mathbb{E}[X \mid \mathcal{G}_2] \mid \mathcal{G}_1\bigr] = \mathbb{E}[X \mid \mathcal{G}_1].$$

Conditioning on less information dominates: the cruder $\sigma$-algebra wins.
:::

:::{prf:proof}
:class: dropdown
:enumerated: false

Let $Z_2 = \mathbb{E}[X \mid \mathcal{G}_2]$ and set $Z_1 = \mathbb{E}[Z_2 \mid \mathcal{G}_1]$. We must verify that $Z_1$ satisfies the defining property of $\mathbb{E}[X \mid \mathcal{G}_1]$: for all $A \in \mathcal{G}_1$,

$$\int_A Z_1 \, d\mathbb{P} = \int_A X \, d\mathbb{P}.$$

Since $A \in \mathcal{G}_1 \subseteq \mathcal{G}_2$, the defining property of $Z_2$ gives $\int_A Z_2 \, d\mathbb{P} = \int_A X \, d\mathbb{P}$. The defining property of $Z_1$ (with $A \in \mathcal{G}_1$) gives $\int_A Z_1 \, d\mathbb{P} = \int_A Z_2 \, d\mathbb{P}$. Combining gives the result. $\square$
:::

In financial terms, the tower property says: the expected payoff of a derivative, given today's information, equals the expectation of tomorrow's conditional expectation. The market cannot do better by waiting — all nested expectations collapse to the coarsest one.

---

## Martingales

The martingale is the central concept of mathematical finance. It formalises the notion of a **fair game**: one in which your expected future wealth equals your current wealth, regardless of the information you have.

:::{prf:definition} Martingale
:label: def-martingale

An adapted, integrable process $(M_t)_{t \geq 0}$ on $(\Omega, \mathcal{F}, \mathbb{P}, \mathbb{F})$ is a **martingale** if

$$\mathbb{E}[M_t \mid \mathcal{F}_s] = M_s \quad \text{for all } 0 \leq s \leq t.$$

It is a **supermartingale** if $\mathbb{E}[M_t \mid \mathcal{F}_s] \leq M_s$ (the game favours the house), and a **submartingale** if $\mathbb{E}[M_t \mid \mathcal{F}_s] \geq M_s$ (the game favours the player).
:::

Brownian motion is the canonical example of a martingale (and we will see this in the next section). A discounted stock price under the risk-neutral measure is also a martingale — this is the content of the **Fundamental Theorem of Asset Pricing**.

:::{prf:theorem} Optional Stopping Theorem (Doob)
:label: thm-ost

Let $(M_t)_{t \in [0,T]}$ be a martingale and $\tau$ a stopping time with $\tau \leq T$ almost surely. Then

$$\mathbb{E}[M_\tau] = \mathbb{E}[M_0].$$
:::

:::{prf:proof}
:class: dropdown
:enumerated: false

For a discrete-time martingale with bounded stopping time $\tau \leq n$, we have the telescoping decomposition:

$$M_\tau - M_0 = \sum_{k=1}^n (M_k - M_{k-1}) \, \mathbf{1}_{\tau \geq k}.$$

Since $\{\tau \geq k\} = \{\tau \leq k-1\}^c \in \mathcal{F}_{k-1}$, and $\mathbb{E}[M_k - M_{k-1} \mid \mathcal{F}_{k-1}] = 0$ by the martingale property, each term has expectation zero. Summing, $\mathbb{E}[M_\tau - M_0] = 0$. The continuous-time result follows by approximation. $\square$
:::

The Optional Stopping Theorem is the mathematical formalisation of a simple truth: you cannot beat a fair game by clever timing. No stopping rule applied to a martingale changes the expected value. This rules out "doubling strategies" and other betting schemes that rely on timing alone.

---

## Change of Measure

One of the most powerful tools in mathematical finance is the ability to **change the probability measure**. Different measures correspond to different perspectives on the same random outcomes: the physical measure $\mathbb{P}$ describes the real-world dynamics; the **risk-neutral measure** $\mathbb{Q}$ is the one under which all discounted asset prices are martingales, and under which derivatives are priced.

:::{prf:definition} Equivalent Measures and Radon–Nikodym Derivative
:label: def-radon-nikodym

Two probability measures $\mathbb{P}$ and $\mathbb{Q}$ on $(\Omega, \mathcal{F})$ are **equivalent** ($\mathbb{P} \sim \mathbb{Q}$) if they have the same null sets: $\mathbb{P}(A) = 0 \iff \mathbb{Q}(A) = 0$. When $\mathbb{Q} \ll \mathbb{P}$ (absolute continuity), the **Radon–Nikodym theorem** guarantees the existence of a non-negative random variable $Z = d\mathbb{Q}/d\mathbb{P}$ such that

$$\mathbb{Q}(A) = \int_A Z \, d\mathbb{P} \quad \text{for all } A \in \mathcal{F}.$$

The process $Z_t = \mathbb{E}^\mathbb{P}[Z \mid \mathcal{F}_t]$ is the **density process**, and it is a $(\mathbb{P}, \mathbb{F})$-martingale.
:::

:::{prf:theorem} Girsanov's Theorem
:label: thm-girsanov

Let $(\Omega, \mathcal{F}, \mathbb{P})$ carry a standard Brownian motion $(W_t)_{t \in [0,T]}$ and let $(\theta_t)$ be an adapted process satisfying the **Novikov condition**:

$$\mathbb{E}^\mathbb{P}\!\left[\exp\!\left(\frac{1}{2} \int_0^T \theta_t^2 \, dt\right)\right] < \infty.$$

Define the measure $\mathbb{Q}$ via the Radon–Nikodym derivative:

$$\frac{d\mathbb{Q}}{d\mathbb{P}}\bigg|_{\mathcal{F}_T} = \exp\!\left(-\int_0^T \theta_t \, dW_t - \frac{1}{2}\int_0^T \theta_t^2 \, dt\right).$$

Then $\mathbb{Q}$ is a probability measure equivalent to $\mathbb{P}$, and the process

$$\tilde{W}_t = W_t + \int_0^t \theta_s \, ds$$

is a standard Brownian motion under $\mathbb{Q}$.
:::

In the Black-Scholes model, $\theta_t = (\mu - r)/\sigma$ is the **Sharpe ratio** (market price of risk). Girsanov's theorem transforms the real-world Brownian motion $W_t$ into the risk-neutral Brownian motion $\tilde{W}_t$, removing the drift $\mu - r$ from asset dynamics. Derivative pricing under the risk-neutral measure then becomes:

$$V_0 = e^{-rT} \mathbb{E}^{\mathbb{Q}}[g(S_T)]$$

for any payoff function $g$. The change of measure is not a modelling assumption — it is a consequence of no-arbitrage.

---

## Arbitrage and Self-Financing Strategies

Before stating the central theorem, we need to make precise what we mean by *arbitrage* — the concept that the entire pricing theory is built to exclude.

A trading strategy is a process $(\phi_t)$ describing how many units of each asset we hold at time $t$. For the strategy to be implementable, it must be **predictable** (we choose positions based on currently available information, not the future). A strategy is **self-financing** if changes in portfolio value come solely from price movements — no cash is injected or withdrawn.

:::{prf:definition} Self-Financing Strategy
:label: def-self-financing

A predictable process $\phi = (\phi_t^0, \phi_t^1, \ldots, \phi_t^N)_{t \in [0,T]}$ representing holdings in a riskless bond and $N$ risky assets is **self-financing** if the value process

$$V_t(\phi) = \phi_t^0 B_t + \sum_{i=1}^N \phi_t^i S_t^i$$

satisfies

$$dV_t(\phi) = \phi_t^0 \, dB_t + \sum_{i=1}^N \phi_t^i \, dS_t^i.$$

Equivalently, in discounted terms (with $\tilde{S}_t^i = S_t^i / B_t$ and $\tilde{V}_t = V_t / B_t$):

$$\tilde{V}_t(\phi) = \tilde{V}_0(\phi) + \sum_{i=1}^N \int_0^t \phi_u^i \, d\tilde{S}_u^i.$$
:::

The self-financing condition formalises the intuition that all gains come from trading, not from external funding. This is the right constraint for pricing: a replicating portfolio must be self-financing, otherwise we could cheat by injecting cash to match any payoff.

:::{prf:definition} Arbitrage
:label: def-arbitrage

An **arbitrage opportunity** is a self-financing strategy $\phi$ such that:
1. $V_0(\phi) = 0$ (zero initial cost),
2. $\mathbb{P}(V_T(\phi) \geq 0) = 1$ (no risk of loss),
3. $\mathbb{P}(V_T(\phi) > 0) > 0$ (positive probability of profit).

A market is **arbitrage-free** if no such strategy exists.
:::

Arbitrage is "something from nothing" — a riskless profit requiring no investment. The no-arbitrage condition is the minimal rationality assumption in financial modelling: if arbitrage existed, rational agents would exploit it until prices adjusted. All of derivative pricing rests on this single assumption.

---

## The Fundamental Theorem of Asset Pricing

Everything in this chapter flows into the following central result of mathematical finance.

:::{prf:theorem} Fundamental Theorem of Asset Pricing (Harrison–Pliska, 1981)
:label: thm-ftap

Consider a financial market on $(\Omega, \mathcal{F}, \mathbb{P}, \mathbb{F})$. Let $S_t$ denote the discounted price process of the risky assets. Then:

1. The market is **arbitrage-free** if and only if there exists at least one equivalent martingale measure $\mathbb{Q} \sim \mathbb{P}$ under which $S_t$ is a martingale.
2. The market is **complete** (every payoff is replicable) if and only if the equivalent martingale measure is unique.
:::

This theorem is the cornerstone of derivative pricing. The two clauses are remarkable:

- *No arbitrage* is equivalent to the existence of a pricing measure. This means that risk-neutral pricing is not a modelling choice — it is forced by the absence of free money.
- *Completeness* pins down the measure uniquely. In an incomplete market (e.g. with stochastic volatility or jumps), there is a family of equivalent martingale measures, and derivative prices are not uniquely determined by no-arbitrage alone.

The connection between market completeness and neural networks — which is the subject of the chapter on [Market Completeness and Neural Networks](3_market_completeness.md) — is one of the most elegant structural observations in quantitative finance.
