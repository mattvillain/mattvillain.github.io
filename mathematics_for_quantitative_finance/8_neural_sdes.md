# Neural Stochastic Differential Equations

What if the drift and diffusion of a stochastic differential equation were neural networks?

This is the **neural SDE** — a model that marries the structural guarantees of stochastic calculus (no-arbitrage, martingale representation, convergence theorems) with the flexibility of deep learning. The drift and diffusion coefficients, instead of being prescribed by parametric formulas (constant volatility in Black-Scholes, mean-reverting drift in Heston), are **learned from data**. The result is a model class that is both theoretically grounded and expressive enough to capture the complexity of real financial markets.

Neural SDEs sit at the intersection of several active research areas: Neural ODEs (Chen et al., 2018), generative modelling via diffusion processes (Song & Ermon, 2019; Ho et al., 2020), and classical stochastic finance. This chapter introduces them from the financial perspective.

---

## Definition and Existence

:::{prf:definition} Neural SDE
:label: def-neural-sde

A **Neural SDE** is a stochastic differential equation of the form:

$$dX_t = f_\theta(t, X_t) \, dt + g_\phi(t, X_t) \, dW_t, \quad X_0 = x_0$$

where $f_\theta: [0,T] \times \mathbb{R}^d \to \mathbb{R}^d$ (the **drift network**) and $g_\phi: [0,T] \times \mathbb{R}^d \to \mathbb{R}^{d \times m}$ (the **diffusion network**) are neural networks with parameters $\theta$ and $\phi$ respectively, and $(W_t)$ is an $m$-dimensional Brownian motion.
:::

The classical existence and uniqueness theory for SDEs requires the coefficients to satisfy Lipschitz and linear growth conditions. Neural networks with bounded weights and Lipschitz activations (e.g. tanh, sigmoid, or clipped ReLU) satisfy these conditions, so we can invoke the standard theory.

:::{prf:theorem} Existence and Uniqueness for Neural SDEs
:label: thm-neural-sde-existence

If $f_\theta$ and $g_\phi$ are globally Lipschitz in $x$ uniformly in $t$ (which holds when the networks use Lipschitz activations and have bounded weights), then the neural SDE has a unique strong solution $(X_t)_{t \in [0,T]}$ with:

$$\mathbb{E}\!\left[\sup_{t \in [0,T]} |X_t|^2\right] < \infty.$$
:::

In practice, the Lipschitz condition is often not enforced explicitly during training — networks with ReLU activations are only locally Lipschitz. Empirically, training remains stable for moderate time horizons, but the theoretical guarantees are weaker. **Spectral normalisation** of the weight matrices is one way to enforce global Lipschitz constraints while retaining expressiveness.

---

## Simulation and Training

### Forward Simulation

Simulating a neural SDE is exactly the **Euler-Maruyama scheme** from the [Numerical Methods](5_numerical_methods.md) chapter, with the neural networks evaluated at each step:

$$X_{t_{k+1}} = X_{t_k} + f_\theta(t_k, X_{t_k}) \cdot h + g_\phi(t_k, X_{t_k}) \cdot \sqrt{h} \, Z_{k+1}$$

where $h$ is the time step and $Z_{k+1} \sim \mathcal{N}(0, I_m)$. Each forward pass through the neural SDE is a sequence of neural network evaluations interleaved with random noise injections.

### Training via Backpropagation

The parameters $(\theta, \phi)$ are learned by minimising a loss function that depends on the terminal state $X_T$ (or the entire path). The gradient is computed by **backpropagation through the Euler-Maruyama steps** — differentiating through the discrete simulation.

:::{prf:definition} Neural SDE Training
:label: def-nsde-training

Given a loss function $\mathcal{L}(\theta, \phi) = \mathbb{E}[\ell(X_T^{\theta,\phi})]$ (or a path-dependent loss), the parameters are updated via stochastic gradient descent:

$$\theta \leftarrow \theta - \eta \, \nabla_\theta \mathcal{L}, \quad \phi \leftarrow \phi - \eta \, \nabla_\phi \mathcal{L}$$

where gradients are estimated by simulating $M$ paths, computing the loss, and backpropagating through all $n$ Euler-Maruyama steps.
:::

The computational graph has depth $n$ (the number of time steps), so memory scales linearly with $n$. For very fine discretisations, the **adjoint method** offers a memory-efficient alternative.

### The Adjoint Method

:::{prf:theorem} Continuous Adjoint for Neural SDEs (Li et al., 2020)
:label: thm-adjoint-sde

The gradient of $\mathcal{L} = \mathbb{E}[\ell(X_T)]$ with respect to the drift parameters $\theta$ can be computed by solving a **backward SDE** (the adjoint equation):

$$da_t = -a_t \cdot \nabla_x f_\theta(t, X_t) \, dt - a_t \cdot \nabla_x g_\phi(t, X_t) \, dW_t$$

initialised at $a_T = \nabla_x \ell(X_T)$, where $a_t$ is the adjoint state. The parameter gradient is then:

$$\nabla_\theta \mathcal{L} = \int_0^T a_t \cdot \nabla_\theta f_\theta(t, X_t) \, dt.$$
:::

The adjoint method computes exact gradients with $O(1)$ memory (independent of the number of time steps), at the cost of solving a backward SDE. This is the stochastic generalisation of the Neural ODE adjoint (Chen et al., 2018). In practice, the choice between discrete backpropagation and the continuous adjoint depends on the step size and the desired accuracy.

---

## Applications in Finance

### Model Calibration

The most direct application is **calibration**: fitting the neural SDE to match observed market prices.

:::{prf:definition} Neural SDE Calibration
:label: def-nsde-calibration

Given observed option prices $\{V_{\text{mkt}}(K_j, T_j)\}_{j=1}^J$ for various strikes $K_j$ and maturities $T_j$, the calibration problem is:

$$\min_{\theta, \phi} \sum_{j=1}^J \left| V_{\text{model}}^{\theta, \phi}(K_j, T_j) - V_{\text{mkt}}(K_j, T_j) \right|^2$$

where $V_{\text{model}}^{\theta,\phi}(K, T) = e^{-rT} \mathbb{E}^\mathbb{Q}[(X_T^{\theta,\phi} - K)^+]$ is the option price under the neural SDE, estimated by Monte Carlo.
:::

Classical parametric models (Black-Scholes, Heston, SABR) have fixed functional forms for the volatility and drift, with a handful of parameters. They calibrate quickly but cannot match the full implied volatility surface simultaneously — the **calibration error** is non-zero and structured. Neural SDEs, with their much larger parameter space, can fit the entire surface to machine precision. The cost is training time and the risk of overfitting.

A practical approach is **regularised calibration**: add penalty terms that encourage the learned dynamics to be smooth and to satisfy desirable properties (e.g. non-negative volatility, martingale property for the discounted price under $\mathbb{Q}$).

### Volatility Surface Learning

Closely related to calibration is the problem of learning the **local volatility surface** $\sigma(t, S)$ from options data. In the Dupire framework, the local volatility is uniquely determined by the observed call prices via the **Dupire formula**:

$$\sigma^2(T, K) = \frac{\partial C / \partial T + rK \, \partial C / \partial K}{\frac{1}{2}K^2 \, \partial^2 C / \partial K^2}.$$

A neural network $\sigma_\theta(t, s)$ can be trained to satisfy this formula by fitting it to the observed option surface, producing a smooth, arbitrage-free local volatility surface. This avoids the numerical instabilities that plague direct Dupire inversion.

### Market Simulation and Scenario Generation

Neural SDEs are natural **generative models** for financial time series. Once trained on historical data, they can generate realistic synthetic market paths for:

- **Stress testing**: Simulate extreme but plausible market scenarios.
- **Risk management**: Estimate VaR and CVaR by simulating many portfolio paths.
- **Data augmentation**: Generate additional training data for downstream ML models.

The quality of generated paths can be evaluated by comparing statistical properties (autocorrelation of returns, volatility clustering, heavy tails, leverage effect) with historical data.

---

## Connection to Generative Diffusion Models

There is a deep connection between neural SDEs and the **score-based diffusion models** that have revolutionised generative AI (images, audio, video).

A diffusion generative model works in two phases:
1. **Forward process**: Gradually add noise to data, following a fixed SDE $dX_t = f(t, X_t) \, dt + g(t) \, dW_t$ (typically an Ornstein-Uhlenbeck process that drives data toward a Gaussian).
2. **Reverse process**: Learn the time-reversal, which is itself an SDE $dX_t = [f(t, X_t) - g(t)^2 \nabla_x \log p_t(X_t)] \, dt + g(t) \, d\bar{W}_t$, where $\nabla_x \log p_t$ (the **score function**) is estimated by a neural network.

This is precisely a neural SDE: the reverse-time drift includes a learned component (the score network) and the process is simulated via Euler-Maruyama.

:::{prf:remark}
:label: rmk-diffusion-finance

Diffusion generative models have been applied to financial data generation (e.g. generating realistic order book states, yield curve dynamics, or multivariate return distributions). The advantage over GANs is stable training and the availability of exact likelihood computation via the continuous normalizing flow / Fokker-Planck connection.

From a categorical perspective, neural SDEs are **morphisms in a stochastic Para category**: the parameter objects are the network weights, and the morphisms are parameterised stochastic kernels. Training is optimisation over the parameter space. This connects to the broader theme of this book — finding a common categorical language for neural networks and probabilistic financial models.
:::

---

## Challenges and Open Problems

Several challenges remain in the practical deployment of neural SDEs for finance:

1. **Arbitrage-freeness.** A calibrated neural SDE should produce arbitrage-free prices. This requires the discounted price process to be a martingale under $\mathbb{Q}$, which places constraints on the drift network. Enforcing this during training is an active research problem.

2. **Path regularity.** Neural SDEs can produce paths with unrealistic properties (e.g. excessively smooth or rough) depending on the network architecture. Matching the empirical roughness of financial data (which often exhibits $H \approx 0.1$ Hurst exponent, consistent with rough volatility models) may require specialised architectures.

3. **Interpretability.** A calibrated neural SDE fits the market, but what has it learned? Extracting economic insights from the learned drift and diffusion functions is non-trivial. Techniques from interpretability (e.g. Shapley-based feature importance, as in our [ICAIF paper](https://arxiv.org/pdf/2210.02176) on time-series feature importance) can help.

4. **Computational cost.** Training requires simulating many paths and backpropagating through all time steps. For high-dimensional problems (e.g. yield curve models with 20+ factors), this becomes expensive.
