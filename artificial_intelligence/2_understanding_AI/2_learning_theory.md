# Learning Theory

We have defined neural networks and shown they can approximate any continuous function. But approximation is not learning. A network that memorises its training data but fails on unseen inputs has not learned anything useful. **Learning theory** asks: under what conditions can a model generalise from finite training data to the underlying distribution?

This is a deep question that sits at the intersection of statistics, optimisation, and combinatorics. The answers are both comforting (learning is possible under mild conditions) and sobering (there are fundamental limits that no amount of data or compute can overcome).

---

## The Learning Problem

:::{prf:definition} Supervised Learning Problem
:label: def-supervised-learning

A **supervised learning problem** consists of:
- A domain $\mathcal{X}$ (input space) and a label set $\mathcal{Y}$ (output space).
- An unknown distribution $\mathcal{D}$ over $\mathcal{X} \times \mathcal{Y}$.
- A training set $S = \{(x_i, y_i)\}_{i=1}^m$ drawn i.i.d. from $\mathcal{D}$.
- A hypothesis class $\mathcal{H} \subseteq \{h: \mathcal{X} \to \mathcal{Y}\}$.
- A loss function $\ell: \mathcal{Y} \times \mathcal{Y} \to [0, \infty)$.

The **population risk** (true error) of a hypothesis $h$ is $R(h) = \mathbb{E}_{(x,y) \sim \mathcal{D}}[\ell(h(x), y)]$.

The **empirical risk** (training error) is $\hat{R}_S(h) = \frac{1}{m}\sum_{i=1}^m \ell(h(x_i), y_i)$.

The goal is to find $h \in \mathcal{H}$ with small population risk, using only the training set $S$.
:::

The fundamental tension: we can compute $\hat{R}_S(h)$ but we want to minimise $R(h)$. The gap $R(h) - \hat{R}_S(h)$ is the **generalisation gap**, and bounding it is the central task of learning theory.

---

## PAC Learning

The **Probably Approximately Correct** (PAC) framework, introduced by Leslie Valiant (1984), formalises what it means for a hypothesis class to be learnable.

:::{prf:definition} PAC Learnability
:label: def-pac

A hypothesis class $\mathcal{H}$ is **PAC learnable** if there exists an algorithm $\mathcal{A}$ and a function $m_\mathcal{H}: (0,1)^2 \to \mathbb{N}$ such that: for every $\varepsilon, \delta \in (0,1)$ and every distribution $\mathcal{D}$ over $\mathcal{X} \times \mathcal{Y}$, if $m \geq m_\mathcal{H}(\varepsilon, \delta)$ then with probability at least $1 - \delta$ over the draw of $S \sim \mathcal{D}^m$:

$$R(\mathcal{A}(S)) \leq \min_{h \in \mathcal{H}} R(h) + \varepsilon.$$

The function $m_\mathcal{H}(\varepsilon, \delta)$ is the **sample complexity**: the number of examples needed to learn within accuracy $\varepsilon$ with confidence $1 - \delta$.
:::

PAC learning asks for two things simultaneously: the error should be small (approximately correct) and this should hold with high probability (probably). The "agnostic" version above allows the best hypothesis in $\mathcal{H}$ to have nonzero error — the class may not contain the true function.

For finite hypothesis classes, a simple union bound gives:

:::{prf:theorem} PAC Bound for Finite Classes
:label: thm-pac-finite

If $|\mathcal{H}| < \infty$ and $\ell \in [0,1]$, then $\mathcal{H}$ is PAC learnable with sample complexity:

$$m_\mathcal{H}(\varepsilon, \delta) = \left\lceil \frac{\log(2|\mathcal{H}|/\delta)}{2\varepsilon^2} \right\rceil.$$
:::

:::{prf:proof}
:class: dropdown
:enumerated: false

By Hoeffding's inequality, for any fixed $h \in \mathcal{H}$: $\mathbb{P}(|R(h) - \hat{R}_S(h)| > \varepsilon) \leq 2e^{-2m\varepsilon^2}$.

By the union bound over all $h \in \mathcal{H}$: $\mathbb{P}(\exists h \in \mathcal{H}: |R(h) - \hat{R}_S(h)| > \varepsilon) \leq 2|\mathcal{H}|e^{-2m\varepsilon^2}$.

Setting this $\leq \delta$ and solving for $m$ gives $m \geq \log(2|\mathcal{H}|/\delta) / (2\varepsilon^2)$. $\square$
:::

The sample complexity scales as $\log |\mathcal{H}|$ — logarithmically in the size of the hypothesis class. This is good news: even large (but finite) classes are learnable with manageable amounts of data.

---

## VC Dimension

For infinite hypothesis classes (like neural networks), the cardinality $|\mathcal{H}|$ is infinite and the union bound fails. We need a more refined measure of complexity: the **VC dimension**.

:::{prf:definition} Shattering and VC Dimension
:label: def-vc-dimension

A hypothesis class $\mathcal{H} \subseteq \{h: \mathcal{X} \to \{0,1\}\}$ **shatters** a set $C = \{x_1, \ldots, x_d\} \subset \mathcal{X}$ if for every labelling $(y_1, \ldots, y_d) \in \{0,1\}^d$, there exists $h \in \mathcal{H}$ with $h(x_i) = y_i$ for all $i$.

The **Vapnik–Chervonenkis (VC) dimension** $\text{VCdim}(\mathcal{H})$ is the size of the largest set shattered by $\mathcal{H}$:

$$\text{VCdim}(\mathcal{H}) = \max\{d : \exists C \subset \mathcal{X},\, |C| = d,\, \mathcal{H} \text{ shatters } C\}.$$
:::

The VC dimension measures the **expressiveness** of a hypothesis class: how many points it can classify in all possible ways. A class with VC dimension $d$ can fit any labelling of $d$ points but not $d+1$.

**Examples:**
- Linear classifiers in $\mathbb{R}^n$: $\text{VCdim} = n + 1$. A line in $\mathbb{R}^2$ can shatter 3 points (in general position) but not 4.
- Intervals $[a, b] \subset \mathbb{R}$: $\text{VCdim} = 2$.
- ReLU networks with $W$ parameters: $\text{VCdim} = O(WL \log(WL))$ where $L$ is the depth (Bartlett et al., 2019). This grows with the number of parameters but is not simply $W$.

:::{prf:theorem} Fundamental Theorem of Statistical Learning
:label: thm-fundamental-sl

For binary classification with 0-1 loss, the following are equivalent:

1. $\mathcal{H}$ has finite VC dimension.
2. $\mathcal{H}$ is PAC learnable.
3. **Uniform convergence** holds: $\sup_{h \in \mathcal{H}} |R(h) - \hat{R}_S(h)| \to 0$ as $m \to \infty$.

Moreover, the sample complexity satisfies $m_\mathcal{H}(\varepsilon, \delta) = \Theta\!\left(\frac{d + \log(1/\delta)}{\varepsilon^2}\right)$ where $d = \text{VCdim}(\mathcal{H})$.
:::

This is remarkable: the VC dimension is the **one number** that determines whether a class is learnable. The sample complexity grows linearly in the VC dimension — replacing $\log|\mathcal{H}|$ for finite classes with $d$ for infinite ones.

---

## The Bias-Variance Tradeoff

A more operational perspective on generalisation comes from the **bias-variance decomposition**.

:::{prf:theorem} Bias-Variance Decomposition
:label: thm-bias-variance

For squared loss $\ell(y, \hat{y}) = (y - \hat{y})^2$ and a learning algorithm $\mathcal{A}$ that maps training sets $S$ to hypotheses $h_S$, the expected population risk decomposes as:

$$\mathbb{E}_S[R(h_S)] = \text{Bias}^2 + \text{Variance} + \text{Noise}$$

where (at a fixed input $x$):
- $\text{Bias}^2 = (\mathbb{E}_S[h_S(x)] - f^*(x))^2$ measures how far the average prediction is from the truth,
- $\text{Variance} = \mathbb{E}_S[(h_S(x) - \mathbb{E}_S[h_S(x)])^2]$ measures how much predictions vary across training sets,
- $\text{Noise} = \mathbb{E}[(y - f^*(x))^2]$ is the irreducible error (label noise).
:::

Simple models (small $\mathcal{H}$) have high bias but low variance. Complex models (large $\mathcal{H}$) have low bias but high variance. Classical wisdom says to choose complexity to minimise the sum — but as we discussed in the neural networks chapter, the **double descent** phenomenon shows that very complex models can escape this tradeoff entirely.

---

## Generalisation in Deep Learning

The classical theory — VC dimension, Rademacher complexity, PAC bounds — predicts that overparameterised neural networks should generalise poorly. A network with millions of parameters has enormous VC dimension and can memorise random labels. Yet in practice, these networks generalise remarkably well.

This is the **generalisation puzzle** of deep learning, and it remains one of the most important open problems in the field. Several complementary explanations have been proposed:

:::{prf:definition} Rademacher Complexity
:label: def-rademacher

The **empirical Rademacher complexity** of $\mathcal{H}$ on a sample $S = \{x_1, \ldots, x_m\}$ is:

$$\hat{\mathfrak{R}}_S(\mathcal{H}) = \mathbb{E}_\sigma\!\left[\sup_{h \in \mathcal{H}} \frac{1}{m}\sum_{i=1}^m \sigma_i h(x_i)\right]$$

where $\sigma_1, \ldots, \sigma_m$ are i.i.d. Rademacher random variables ($\pm 1$ with equal probability).
:::

Rademacher complexity measures how well the class can correlate with random noise. A tighter generalisation bound is:

$$R(h) \leq \hat{R}_S(h) + 2\hat{\mathfrak{R}}_S(\mathcal{H}) + \sqrt{\frac{\log(1/\delta)}{2m}}$$

with probability $1 - \delta$. Unlike VC bounds, Rademacher complexity is data-dependent and can be smaller for structured data distributions.

**Other approaches to explaining deep learning generalisation:**

- **Norm-based bounds**: The generalisation gap depends not on the number of parameters but on the **norms** of the weight matrices. Networks with small spectral norms generalise well regardless of width (Bartlett et al., 2017).
- **PAC-Bayes**: Bounds the generalisation gap in terms of the KL divergence between the learned posterior and a prior over hypotheses. Overparameterised networks trained with SGD remain close to initialisation in a KL sense.
- **Algorithmic stability**: SGD on smooth losses is *uniformly stable* — replacing one training example changes the output by at most $O(1/m)$ — which directly implies generalisation (Hardt et al., 2016).
- **Implicit regularisation**: SGD, even without explicit regularisation, preferentially converges to low-complexity solutions. For linear models, it finds the minimum-norm interpolant; for deep networks, the implicit bias is less understood but empirically strong.

The resolution to the generalisation puzzle likely involves all of these perspectives simultaneously. The effective complexity of the learned function — as opposed to the complexity of the hypothesis class — is what matters for generalisation.
