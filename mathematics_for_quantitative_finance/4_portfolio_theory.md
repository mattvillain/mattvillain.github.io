# Portfolio Theory

We have priced individual derivatives. But a trader does not hold a single option — they hold a *portfolio*. How should one choose among portfolios? Which combinations of assets are "optimal," and in what sense?

This question was first answered rigorously by Harry Markowitz in 1952, and the framework he introduced — **mean-variance optimisation** — remains the foundation of modern portfolio theory. It is elegant, tractable, and deeply connected to ideas in machine learning: the efficient frontier is, at its core, a bias-variance tradeoff, and the CAPM is a linear regression.

---

## Returns and Risk

We begin with the basic ingredients. Suppose we have $N$ risky assets with random returns $R = (R_1, \ldots, R_N)^\top$ over one period. A **portfolio** is specified by a weight vector $w = (w_1, \ldots, w_N)^\top$ where $w_i$ is the fraction of wealth invested in asset $i$.

:::{prf:definition} Portfolio Return and Risk
:label: def-portfolio-return

Given asset returns $R \in \mathbb{R}^N$ with mean vector $\boldsymbol{\mu} = \mathbb{E}[R] \in \mathbb{R}^N$ and covariance matrix $\Sigma = \operatorname{Cov}(R) \in \mathbb{R}^{N \times N}$ (positive definite), a portfolio with weights $w \in \mathbb{R}^N$ satisfying $\mathbf{1}^\top w = 1$ has:

- **Expected return**: $\mu_p = w^\top \boldsymbol{\mu}$,
- **Variance (risk)**: $\sigma_p^2 = w^\top \Sigma w$,
- **Standard deviation**: $\sigma_p = \sqrt{w^\top \Sigma w}$.
:::

The covariance matrix $\Sigma$ encodes how assets co-move. Its off-diagonal entries are what make portfolio theory non-trivial: by combining negatively correlated assets, one can achieve lower variance than any individual asset — this is the mathematics of **diversification**.

---

## Mean-Variance Optimisation

Markowitz's key insight was to frame portfolio selection as a **constrained optimisation problem**: among all portfolios achieving a given expected return, find the one with minimum risk.

:::{prf:definition} Markowitz Mean-Variance Problem
:label: def-mv-problem

The **minimum-variance portfolio** achieving target return $\mu_{\text{target}}$ is the solution to:

$$\min_{w \in \mathbb{R}^N} \; w^\top \Sigma w \quad \text{subject to} \quad w^\top \boldsymbol{\mu} = \mu_{\text{target}}, \quad \mathbf{1}^\top w = 1.$$
:::

This is a convex quadratic program with two linear equality constraints. It has a closed-form solution via the method of Lagrange multipliers.

:::{prf:theorem} Efficient Frontier
:label: thm-efficient-frontier

The set of minimum-variance portfolios, as $\mu_{\text{target}}$ varies over $\mathbb{R}$, traces a **parabola** in the $(\sigma_p^2, \mu_p)$ plane (or equivalently, a hyperbola in the $(\sigma_p, \mu_p)$ plane). The optimal weights are:

$$w^*(\mu_{\text{target}}) = \Sigma^{-1} \begin{pmatrix} \boldsymbol{\mu} & \mathbf{1} \end{pmatrix} \begin{pmatrix} \boldsymbol{\mu}^\top \Sigma^{-1} \boldsymbol{\mu} & \boldsymbol{\mu}^\top \Sigma^{-1} \mathbf{1} \\ \mathbf{1}^\top \Sigma^{-1} \boldsymbol{\mu} & \mathbf{1}^\top \Sigma^{-1} \mathbf{1} \end{pmatrix}^{-1} \begin{pmatrix} \mu_{\text{target}} \\ 1 \end{pmatrix}.$$

The minimum variance achieved is a quadratic function of $\mu_{\text{target}}$:

$$\sigma_p^{2*}(\mu_{\text{target}}) = \frac{C \mu_{\text{target}}^2 - 2B\mu_{\text{target}} + A}{AC - B^2}$$

where $A = \boldsymbol{\mu}^\top \Sigma^{-1} \boldsymbol{\mu}$, $B = \boldsymbol{\mu}^\top \Sigma^{-1} \mathbf{1}$, $C = \mathbf{1}^\top \Sigma^{-1} \mathbf{1}$.
:::

:::{prf:proof}
:class: dropdown
:enumerated: false

Form the Lagrangian:

$$\mathcal{L}(w, \lambda, \gamma) = w^\top \Sigma w - \lambda(w^\top \boldsymbol{\mu} - \mu_{\text{target}}) - \gamma(w^\top \mathbf{1} - 1).$$

First-order conditions: $\nabla_w \mathcal{L} = 2\Sigma w - \lambda \boldsymbol{\mu} - \gamma \mathbf{1} = 0$, giving $w^* = \frac{1}{2}\Sigma^{-1}(\lambda \boldsymbol{\mu} + \gamma \mathbf{1})$.

Substituting into the two constraints $w^{*\top}\boldsymbol{\mu} = \mu_{\text{target}}$ and $w^{*\top}\mathbf{1} = 1$ yields a $2 \times 2$ linear system in $(\lambda, \gamma)$:

$$\frac{1}{2}\begin{pmatrix} A & B \\ B & C \end{pmatrix} \begin{pmatrix} \lambda \\ \gamma \end{pmatrix} = \begin{pmatrix} \mu_{\text{target}} \\ 1 \end{pmatrix}$$

where $A = \boldsymbol{\mu}^\top \Sigma^{-1}\boldsymbol{\mu}$, $B = \boldsymbol{\mu}^\top \Sigma^{-1}\mathbf{1}$, $C = \mathbf{1}^\top \Sigma^{-1}\mathbf{1}$. Solving and substituting back gives the claimed formulas. The second-order condition is automatically satisfied since $\Sigma \succ 0$. $\square$
:::

The **efficient frontier** is the upper branch of the hyperbola in $(\sigma_p, \mu_p)$ space — these are the portfolios that cannot be improved in either dimension without sacrificing the other. Any rational investor should hold a portfolio on this frontier; any portfolio below it is dominated.

The **global minimum variance portfolio** (the vertex of the parabola) has weights $w_{\text{GMV}} = \Sigma^{-1}\mathbf{1} / (\mathbf{1}^\top \Sigma^{-1}\mathbf{1})$ and return $\mu_{\text{GMV}} = B/C$. This is the safest portfolio of risky assets — it ignores expected returns entirely and focuses purely on minimising risk.

:::{prf:remark}
:label: rmk-mv-ml

The Markowitz problem has a direct analogue in machine learning. In regularised regression (e.g. ridge regression), one minimises $\|y - Xw\|^2 + \lambda \|w\|^2$ — a tradeoff between fitting the data (returns) and controlling model complexity (risk). The regularisation parameter $\lambda$ plays the same role as the risk aversion parameter in portfolio theory. The efficient frontier *is* the bias-variance tradeoff curve, viewed from the other side.

More precisely, adding an $\ell_2$ penalty $\lambda \|w\|^2$ to the Markowitz objective — which practitioners call **shrinkage** — is equivalent to adding $\lambda I$ to the covariance matrix $\Sigma$. This stabilises the inverse $\Sigma^{-1}$ and reduces sensitivity to estimation error, exactly as ridge regression stabilises ordinary least squares.
:::

---

## The Capital Asset Pricing Model

The CAPM, developed independently by Sharpe (1964), Lintner (1965), and Mossin (1966), extends the Markowitz framework by adding a riskless asset and deriving the equilibrium implications. It is both a pricing model and a portfolio prescription.

When a riskless asset with return $r_f$ is available, the efficient frontier becomes a **straight line** in $(\sigma_p, \mu_p)$ space — the **Capital Market Line (CML)** — passing through $(0, r_f)$ and tangent to the risky-asset frontier. The tangency portfolio $w_T$ is the unique optimal combination of risky assets that all investors should hold.

:::{prf:theorem} Capital Asset Pricing Model
:label: thm-capm

In equilibrium, under mean-variance preferences and homogeneous expectations, every asset's expected excess return is proportional to its **systematic risk** (beta):

$$\mathbb{E}[R_i] - r_f = \beta_i \left(\mathbb{E}[R_m] - r_f\right)$$

where $R_m$ is the return of the **market portfolio** (the value-weighted portfolio of all risky assets) and

$$\beta_i = \frac{\operatorname{Cov}(R_i, R_m)}{\operatorname{Var}(R_m)}$$

is the **beta** of asset $i$.
:::

:::{prf:proof}
:class: dropdown
:enumerated: false

With a riskless asset, the investor's problem becomes: choose an allocation $w$ among risky assets and a fraction $(1 - \mathbf{1}^\top w)$ in the riskless asset, maximising return for given risk. The portfolio return is $\mu_p = r_f + w^\top(\boldsymbol{\mu} - r_f \mathbf{1})$ and risk is $\sigma_p^2 = w^\top \Sigma w$.

The efficient frontier is now parameterised by the **Sharpe ratio** $(\mu_p - r_f)/\sigma_p$, which is maximised by the **tangency portfolio**:

$$w_T = \frac{\Sigma^{-1}(\boldsymbol{\mu} - r_f \mathbf{1})}{\mathbf{1}^\top \Sigma^{-1}(\boldsymbol{\mu} - r_f \mathbf{1})}.$$

In equilibrium, if all investors have the same expectations and hold mean-variance optimal portfolios, the aggregate demand for risky assets must equal the market supply. This forces $w_T$ to equal the **market portfolio** weights.

Now consider perturbing the market portfolio by adding a small amount $\epsilon$ of asset $i$. The first-order optimality condition (the Sharpe ratio of the perturbed portfolio must not increase) gives:

$$\mathbb{E}[R_i] - r_f = \frac{\operatorname{Cov}(R_i, R_m)}{\operatorname{Var}(R_m)} \left(\mathbb{E}[R_m] - r_f\right) = \beta_i \left(\mathbb{E}[R_m] - r_f\right). \quad \square$$
:::

The CAPM has a beautifully simple interpretation: only **systematic risk** (exposure to the market) is rewarded. **Idiosyncratic risk** — the component of an asset's return uncorrelated with the market — can be diversified away and earns no risk premium.

:::{prf:remark}
:label: rmk-capm-regression

The CAPM equation $R_i - r_f = \alpha_i + \beta_i(R_m - r_f) + \varepsilon_i$ is a **linear regression**. Beta is the regression coefficient, alpha is the intercept (which should be zero if CAPM holds), and the residual $\varepsilon_i$ is the idiosyncratic return. Testing the CAPM reduces to testing whether $\alpha_i = 0$ for all assets — a standard hypothesis testing problem. The Fama-French three-factor model and subsequent extensions are multiple regressions that add additional explanatory variables (size, value, momentum) beyond the market return.
:::

---

## Limitations and the Bridge to Deep Learning

The elegance of Markowitz and CAPM comes at a cost: both assume that the mean vector $\boldsymbol{\mu}$ and covariance matrix $\Sigma$ are **known**. In practice, they must be estimated from data, and this introduces **estimation risk**.

The sensitivity is severe. Small changes in estimated expected returns can cause dramatic changes in optimal portfolio weights. The sample covariance matrix $\hat{\Sigma}$ is singular or near-singular when the number of assets exceeds the number of observations (which is common in modern markets with thousands of tradable instruments). This is the "curse of dimensionality" for portfolios.

Practitioners have developed many remedies:
- **Shrinkage estimators** (Ledoit-Wolf): pull $\hat{\Sigma}$ toward a structured target (e.g. the identity or a factor model).
- **Factor models**: assume returns are driven by a small number of factors, reducing the dimensionality of $\Sigma$.
- **Robust optimisation**: optimise for the worst case over an uncertainty set for $(\boldsymbol{\mu}, \Sigma)$.
- **Bayesian methods** (Black-Litterman): combine market equilibrium priors with investor views.

But all of these are, in some sense, patches on a fundamentally parametric framework. **Deep learning** offers an alternative: learn the optimal portfolio allocation directly from data, bypassing the estimation of $\boldsymbol{\mu}$ and $\Sigma$ entirely. This is the subject of the chapter on [Deep Portfolio Optimisation](9_deep_portfolio_optimization.md).
