# Introduction to Quantitative Methods

Quantitative finance is concerned with modelling financial markets mathematically and studying the properties of objects such as asset price processes, portfolio strategies, derivative pricing, market microstructure, credit rating, and many other aspects of markets.

I find this field beautiful for the same reason I find mathematics generally beautiful: it forces you to be precise about things that are genuinely hard to pin down, like uncertainty, information, and fairness. The theory of pricing is, at its core, a theory of what you can know and what you can hedge.

Here is a breakdown of what we will cover:

1. **Probability for Finance.** The foundations: probability spaces, filtrations, conditional expectation, martingales, and change of measure. These are the building blocks of every result that follows.
2. **Stochastic Calculus.** Brownian motion, the Itô integral, Itô's lemma, Geometric Brownian Motion, and the Black-Scholes PDE. This is where the mathematics of continuous randomness lives.
3. **Market Completeness and Neural Networks.** The precise connection between a complete financial market and a neural network. The Universal Approximation Theorem and the Fundamental Theorem of Asset Pricing are, at the right level of abstraction, the same statement.
4. **Portfolio Theory.** Mean-variance optimisation, the efficient frontier, and the Capital Asset Pricing Model. The foundations of how to combine assets optimally — and why estimation risk makes this harder than it looks.
5. **Numerical Methods.** Monte Carlo simulation, Euler-Maruyama discretisation, and finite differences for PDEs. Most pricing problems require computation, and these are the classical tools.
6. **American Options and Optimal Stopping.** The Snell envelope, the Longstaff-Schwartz algorithm, and the connection to reinforcement learning. American-style exercise turns pricing into an optimal stopping problem.
7. **Deep Hedging.** Learning hedging strategies with neural networks in incomplete markets with frictions. This is where classical finance meets modern deep learning most directly.
8. **Neural SDEs.** Neural networks as drift and diffusion coefficients of stochastic differential equations. Applications to model calibration, volatility surface learning, and market simulation.
9. **Deep Portfolio Optimisation.** End-to-end portfolio construction with neural networks, reinforcement learning for rebalancing, and interpretability via Shapley values.

The first six chapters build the **classical theory**; the last three show how **deep learning** extends and transforms it. The recurring theme is that finance and neural networks share deep structural connections — and understanding one enriches the other.

I will add topics over time. On the horizon: interest rate models, credit risk, and rough volatility.
