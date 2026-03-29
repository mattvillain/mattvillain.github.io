# Interpretability

A neural network takes an input, performs millions of floating-point operations, and produces an output. **Why** did it produce that output? What features of the input mattered? What internal representations did it form? Can we trust the model's prediction, or is it relying on a spurious correlation?

These questions define the field of **interpretability** (also called explainability or XAI). As neural networks are deployed in high-stakes domains — medicine, finance, criminal justice, autonomous systems — the need to understand their decisions becomes not just a scientific curiosity but a practical and regulatory necessity.

I have worked on interpretability from the feature importance angle, particularly using Shapley values for time series data and counterfactual explanations for ReLU networks. This chapter surveys the main approaches.

---

## Feature Importance

The simplest interpretability question is: which input features contributed most to a given prediction?

### Gradient-Based Attribution

:::{prf:definition} Gradient Attribution
:label: def-gradient-attribution

Given a differentiable model $f_\theta: \mathbb{R}^d \to \mathbb{R}$ and an input $x \in \mathbb{R}^d$, the **gradient attribution** assigns importance:

$$\phi_j^{\text{grad}}(x) = \frac{\partial f_\theta(x)}{\partial x_j}$$

to feature $j$. The **gradient $\times$ input** variant is $\phi_j(x) = x_j \cdot \partial f_\theta / \partial x_j$.
:::

Gradient attribution has the advantage of being cheap to compute (one backward pass), but it can be noisy (the gradient may point in a direction that is irrelevant to the prediction) and it captures only **local** sensitivity — how the output changes for infinitesimal perturbations, not for realistic changes.

**Integrated Gradients** (Sundararajan et al., 2017) addresses this by integrating the gradient along a path from a baseline $x'$ (e.g., zero) to the input $x$:

$$\phi_j^{\text{IG}}(x) = (x_j - x_j') \int_0^1 \frac{\partial f_\theta(x' + \alpha(x - x'))}{\partial x_j} \, d\alpha.$$

This satisfies the **completeness axiom**: $\sum_j \phi_j^{\text{IG}}(x) = f_\theta(x) - f_\theta(x')$, ensuring the attributions fully account for the prediction.

### Shapley Values

**Shapley values** (Shapley, 1953) originate in cooperative game theory and provide the unique attribution method satisfying a set of natural axioms (efficiency, symmetry, linearity, null player).

:::{prf:definition} Shapley Value
:label: def-shapley

Given a model $f_\theta$ and an input $x = (x_1, \ldots, x_d)$, the **Shapley value** of feature $j$ is:

$$\phi_j(x) = \sum_{S \subseteq \{1,\ldots,d\} \setminus \{j\}} \frac{|S|!(d-|S|-1)!}{d!} \left[v(S \cup \{j\}) - v(S)\right]$$

where $v(S) = \mathbb{E}[f_\theta(x_S, X_{\bar{S}})]$ is the expected model output when features in $S$ are fixed to their observed values and features outside $S$ are marginalised.
:::

:::{prf:theorem} Shapley Uniqueness
:label: thm-shapley-uniqueness

The Shapley value is the **unique** attribution method satisfying:
1. **Efficiency**: $\sum_j \phi_j = v(\{1,\ldots,d\}) - v(\emptyset)$.
2. **Symmetry**: If features $j$ and $k$ contribute equally to every coalition, $\phi_j = \phi_k$.
3. **Linearity**: $\phi_j[v + w] = \phi_j[v] + \phi_j[w]$.
4. **Null player**: If feature $j$ never changes the output, $\phi_j = 0$.
:::

Computing exact Shapley values requires summing over all $2^{d-1}$ subsets — exponential in the number of features. Efficient approximations are essential.

:::{prf:remark}
:label: rmk-shapley-work

**KernelSHAP** (Lundberg & Lee, 2017) approximates Shapley values via a weighted regression, treating each coalition as a data point. Our work on [Feature Importance for Time Series Data](https://arxiv.org/pdf/2210.02176) (ICAIF 2022) addressed a subtlety that arises when the model takes sequential inputs: treating each time step as an independent "player" violates the Shapley axioms because time-series features are inherently ordered. We proposed methods that respect the temporal structure.

More recently, our [Unified Framework for Provably Efficient Algorithms to Estimate Shapley Values](https://arxiv.org/pdf/2506.05216) (NeurIPS 2025) showed that the two main estimation approaches — matrix-vector and least-squares — can be unified under a single framework, yielding provably efficient algorithms that scale to high-dimensional problems.
:::

---

## Counterfactual Explanations

**Counterfactual explanations** answer the question: "what is the smallest change to the input that would change the model's decision?"

:::{prf:definition} Counterfactual Explanation
:label: def-counterfactual

Given a model $f_\theta$, an input $x$ with prediction $f_\theta(x) = y$, and a target class $y'$, a **counterfactual** $x'$ is a solution to:

$$x' = \arg\min_{x''} \; d(x, x'') \quad \text{subject to} \quad f_\theta(x'') = y'$$

where $d$ is a distance metric (e.g., $\ell_1$, $\ell_2$, or a domain-specific metric). The counterfactual is the **nearest input with a different prediction**.
:::

Counterfactuals are actionable: "your loan was denied, but if your income were $5{,}000$ higher, it would be approved." They provide recourse — a concrete change the individual can make to get a different outcome.

:::{prf:remark}
:label: rmk-pice

For ReLU networks, the input space is partitioned into **linear regions** (polyhedral cells) within which the network computes an affine function. Our work on [PICE: Polyhedral Complex Informed Counterfactual Explanations](https://ojs.aaai.org/index.php/AIES/article/view/31742) (AIES in AAAI, 2025) exploited this piecewise-linear structure to compute **exactly minimal** counterfactuals — not approximate, but provably the closest point with a different decision, by enumerating the relevant polytopes. This is computationally expensive but gives exact guarantees, which is important in high-stakes applications where approximate explanations may be misleading.
:::

---

## Mechanistic Interpretability

**Mechanistic interpretability** goes beyond "which features matter?" to ask: "what algorithms does the network implement internally?" The goal is to reverse-engineer the network's computation into human-understandable components.

### Probing

:::{prf:definition} Linear Probe
:label: def-probe

A **linear probe** tests whether a concept $c$ (e.g., part of speech, sentiment, factual knowledge) is encoded in a model's internal representations. Given hidden states $\{h_i\}$ and concept labels $\{c_i\}$, train a linear classifier $p(c \mid h) = \sigma(w^\top h + b)$. If the probe achieves high accuracy, the concept is **linearly represented** in the hidden state.
:::

Probing has revealed that language models encode syntactic parse trees, semantic roles, factual knowledge, and world models in their internal representations — often in linearly decodable form.

### Circuits and Features

The **circuits** programme (Olah et al., 2020) aims to decompose neural networks into:

- **Features**: Individual neurons or directions in activation space that represent meaningful concepts (curve detectors, entity recognisers, etc.).
- **Circuits**: Small subgraphs of the network that implement specific computations (induction heads for in-context learning, indirect object identification circuits, etc.).

The hypothesis is that neural networks can be understood as compositions of interpretable circuits — much as a computer program can be understood as a composition of functions. If this programme succeeds, it would provide a complete, mechanistic understanding of how models work.

### Superposition

A complication: networks often represent more features than they have neurons, by encoding multiple concepts in **superposition** — different features are assigned nearly orthogonal directions in a shared activation space. **Sparse autoencoders** (Cunningham et al., 2023) can decompose superposed representations into interpretable components by learning a sparse dictionary of features from activation data.

---

## Interpretability and Trust

Interpretability is not just about understanding models — it is about **trusting** them. A model that achieves high accuracy but does so by relying on a spurious correlation (e.g., classifying wolves by the snow in the background) is dangerously fragile. Interpretability tools help identify these failure modes before deployment.

The tension between model complexity and interpretability is real but sometimes overstated. Simple models (linear regression, decision trees) are directly interpretable but may lack the capacity to capture complex patterns. Complex models (deep networks) capture the patterns but resist easy interpretation. Post-hoc interpretability methods (Shapley values, counterfactuals, probing) attempt to bridge this gap by explaining complex models without simplifying them.

Whether current interpretability methods provide *reliable* explanations — or merely plausible-sounding stories — remains an active debate. The field is young and rapidly evolving.
