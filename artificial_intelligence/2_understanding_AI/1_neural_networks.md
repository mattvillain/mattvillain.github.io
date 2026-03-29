# Neural Networks

The neural network is the intellectual powerhouse of artificial intelligence. Not all intelligent systems are neural networks, but most modern intelligent systems have them somewhere in their pipeline.

Neural networks are interesting mathematical objects and there are several ways to define them. Each definition illuminates a different facet of what a network is and what it can do. We will explore four definitions, then prove some fundamental results that characterise their expressive power and limits.

---

## Definitions

### Definition 1: The Feedforward Network

The simplest concrete neural network is the **feedforward** (or **multilayer perceptron**) architecture. It is a function built by composing affine maps and a fixed non-linearity. The simplest valid example is the 1-hidden-layer network:

$$f_\theta(x) = W_2 \, \sigma(W_1 x + b_1) + b_2$$

where $\sigma$ is a non-linear **activation function** applied element-wise, and $\theta = (W_1, b_1, W_2, b_2)$ is the collection of **parameters**. Today the most popular activation is the **Rectified Linear Unit (ReLU)**: $\sigma(t) = \max(0, t)$, though historically sigmoid and tanh were more common.

:::{prf:definition} Feedforward Neural Network
:label: def-feedforward-nn

A **feedforward neural network** with $L$ layers is a function $f_\theta: \mathbb{R}^{d_0} \to \mathbb{R}^{d_L}$ defined by the composition

$$f_\theta(x) = W_L \, \sigma\!\left(W_{L-1} \, \sigma\!\left(\cdots \sigma(W_1 x + b_1) \cdots\right) + b_{L-1}\right) + b_L$$

where $W_\ell \in \mathbb{R}^{d_\ell \times d_{\ell-1}}$ and $b_\ell \in \mathbb{R}^{d_\ell}$ for $\ell = 1, \ldots, L$, and $\sigma: \mathbb{R} \to \mathbb{R}$ is a non-linear activation applied element-wise. The parameter set is $\theta = \{(W_\ell, b_\ell)\}_{\ell=1}^L$ and the **parameter space** is $\Theta = \prod_\ell \mathbb{R}^{d_\ell \times d_{\ell-1}} \times \mathbb{R}^{d_\ell}$.
:::

The inputs $x \in \mathbb{R}^{d_0}$ are the data we process. The parameters $\theta$ are initially chosen at random and then updated during training. The non-linearity $\sigma$ is crucial: without it, the entire network collapses to a single linear map (composition of linear maps is linear), and linear functions are weak approximators. The non-linearity is precisely what gives neural networks their expressive power.

The integer $d_\ell$ is the **width** of layer $\ell$, and $L$ is the **depth**. Both matter for expressiveness, and in different ways, as we will see below.

### Definition 2: Parametrised Universal Approximators

A more abstract and arguably more illuminating definition views a neural network as a **parametrised function class** with universal approximation. Rather than fixing the architecture, we characterise networks by their functional properties.

:::{prf:definition} Parametrised Function
:label: def-param-function

A **parametrised function** is a pair $(f, \Theta)$ where $\Theta$ is a topological space of parameters and $f: \mathcal{X} \times \Theta \to \mathcal{Y}$ is a (measurable) function. We write $f_\theta(x) \coloneqq f(x, \theta)$ to indicate evaluation at a fixed $\theta \in \Theta$.

A parametrised function is a **universal approximator** on a function class $\mathcal{G}$ if for every $g \in \mathcal{G}$ and $\varepsilon > 0$ there exists $\theta \in \Theta$ such that $\sup_x |f_\theta(x) - g(x)| < \varepsilon$.
:::

Under this view, a feedforward network is a parametrised function whose parameter space is the product of weight matrices and bias vectors. The Universal Approximation Theorem (below) establishes that sufficiently wide networks are universal approximators on the class of continuous functions.

One should be careful: this definition is broad. Decision trees, Gaussian processes, and radial basis function networks are also universal approximators. Neural networks are not uniquely defined by approximation capacity alone — they are distinguished by their specific parametric form, and importantly by the dynamics of gradient-based optimisation over $\Theta$.

### Definition 3: Conditional Probability Distributions

In much of the deep learning literature — particularly in generative modelling, language modelling, and probabilistic machine learning — a neural network is presented as a **conditional probability distribution** $p_\theta(y \mid x)$.

:::{prf:definition} Neural Network as Conditional Distribution
:label: def-nn-cpd

A neural network $f_\theta$ defines a conditional probability distribution $p_\theta(y \mid x)$ when its output is passed through a normalisation or sampling layer. Common constructions are:

- **Classification**: $p_\theta(y \mid x) = \operatorname{Softmax}(f_\theta(x))_y$, where $\operatorname{Softmax}(z)_i = e^{z_i} / \sum_j e^{z_j}$.
- **Regression**: $p_\theta(y \mid x) = \mathcal{N}(y;\, f_\theta(x),\, \sigma^2)$ for some variance $\sigma^2$.
- **Autoregressive generation**: $p_\theta(x_1, \ldots, x_T) = \prod_{t=1}^T p_\theta(x_t \mid x_1, \ldots, x_{t-1})$.
:::

This probabilistic lens is extremely productive. Training becomes **maximum likelihood estimation**: given data $\{(x_i, y_i)\}_{i=1}^n$, we maximise

$$\mathcal{L}(\theta) = \sum_{i=1}^n \log p_\theta(y_i \mid x_i).$$

The ubiquitous **cross-entropy loss** is exactly the negative log-likelihood of the categorical distribution — its prevalence in classification is therefore perfectly natural from this perspective, not an ad hoc choice.

The probabilistic view also clarifies the role of **temperature scaling**, **calibration**, and **Bayesian neural networks**, where one places a prior on $\theta$ and computes a posterior distribution over parameters given data.

### Definition 4: A Categorical View

For the categorically-minded, there is an elegant way to organise all of the above. A neural network layer is a **parametrised morphism** in the category $\textbf{Para}(\textbf{Smooth})$, whose objects are smooth manifolds and whose morphisms are smooth maps.

:::{prf:definition} Para Construction
:label: def-para

Given a symmetric monoidal category $(\mathcal{C}, \otimes, I)$, the category $\textbf{Para}(\mathcal{C})$ has the same objects as $\mathcal{C}$, and a morphism from $A$ to $B$ is a pair $(P, f)$ where $P \in \mathcal{C}$ is a **parameter object** and $f: P \otimes A \to B$ is a morphism in $\mathcal{C}$. Composition of $(P_1, f): A \to B$ and $(P_2, g): B \to C$ is:

$$(P_1 \times P_2,\; (p_1, p_2, x) \mapsto g(p_2, f(p_1, x))) : A \to C.$$
:::

Under this definition, a neural network with $L$ layers is a composite of $L$ parametrised morphisms in $\textbf{Para}(\textbf{Smooth})$. The total parameter space is the product $P_1 \times \cdots \times P_L$. Training corresponds to finding a point in this product space that minimises a loss functional.

The categorical formulation makes the **modularity** of neural architectures precise: modules compose, parameter spaces stack, and architectural choices are choices of morphism in a particular category. It also opens the door to replacing $\textbf{Smooth}$ with other categories — for instance, $\textbf{Markov}$ (for stochastic networks) or $\textbf{Vect}_\mathbb{R}$ (for linear networks).

I am biased here, but I think this is the right way to think about neural networks. If you want to read more about category theory, I have a dedicated section.

---

## Theorems

### Universal Approximation

The most fundamental theoretical result about neural networks is that they can approximate any continuous function arbitrarily well, given sufficient width. This was established independently by Cybenko (1989) for sigmoidal activations and by Hornik, Stinchcombe and White (1989) for general squashing functions.

:::{prf:theorem} Universal Approximation Theorem
:label: thm-uat

Let $\sigma: \mathbb{R} \to \mathbb{R}$ be any continuous sigmoidal function (i.e. $\sigma(t) \to 1$ as $t \to +\infty$ and $\sigma(t) \to 0$ as $t \to -\infty$). Let $I_n = [0,1]^n$. Then for any $g \in C(I_n)$ and any $\varepsilon > 0$, there exist $N \in \mathbb{N}$, weights $\alpha_j \in \mathbb{R}$, vectors $y_j \in \mathbb{R}^n$, and biases $\theta_j \in \mathbb{R}$ such that

$$\sup_{x \in I_n} \left| g(x) - \sum_{j=1}^N \alpha_j \, \sigma(y_j \cdot x + \theta_j) \right| < \varepsilon.$$

In other words, finite sums of the form $\sum_{j=1}^N \alpha_j \sigma(y_j \cdot x + \theta_j)$ are **dense** in $C(I_n)$ under the uniform norm.
:::

:::{prf:proof}
:class: dropdown
:enumerated: false

We follow Cybenko's original argument — a beautiful application of functional analysis. Let $\Sigma$ denote the set of all finite sums $\sum_j \alpha_j \sigma(y_j \cdot x + \theta_j)$, and let $\overline{\Sigma}$ be its closure in $C(I_n)$ under the sup-norm $\|\cdot\|_\infty$.

**Step 1.** Suppose for contradiction that $\overline{\Sigma} \neq C(I_n)$. By the **Hahn–Banach theorem**, there exists a nonzero bounded linear functional $L: C(I_n) \to \mathbb{R}$ that vanishes on $\overline{\Sigma}$. By the **Riesz Representation Theorem**, $L$ is represented by a unique finite signed Borel measure $\mu$ on $I_n$:

$$L(h) = \int_{I_n} h(x) \, d\mu(x).$$

**Step 2.** Since $L$ vanishes on all sigmoidal units, for every $y \in \mathbb{R}^n$ and $\theta \in \mathbb{R}$:

$$\int_{I_n} \sigma(y \cdot x + \theta) \, d\mu(x) = 0.$$

Fix $\omega \in S^{n-1}$ and consider $y = \lambda \omega$ for $\lambda > 0$. As $\lambda \to +\infty$, the sigmoidal function $\sigma(\lambda(\omega \cdot x) + \theta)$ converges pointwise to the indicator $\mathbf{1}_{\{x : \omega \cdot x > -\theta/\lambda\}}$. Similarly as $\lambda \to -\infty$ it converges to $\mathbf{1}_{\{x : \omega \cdot x \geq 0\}}$ (up to null sets). By the **Bounded Convergence Theorem**, passing to limits shows that $\mu$ integrates every half-space indicator to zero.

**Step 3.** A signed measure that assigns zero mass to every half-space of the form $\{x : \omega \cdot x \geq c\}$ for all $\omega \in S^{n-1}$ and $c \in \mathbb{R}$ must be zero. Indeed, the algebra generated by half-spaces is the Borel $\sigma$-algebra on $I_n$, so $\mu = 0$.

This contradicts $L \neq 0$. Therefore $\overline{\Sigma} = C(I_n)$. $\square$
:::

The UAT is an **existence theorem**: it guarantees that some set of weights achieves the approximation, but says nothing about how to find them. It also says nothing about the **number of neurons** $N$ required (though tight bounds are known). Two important caveats:

1. **Depth matters.** Certain functions require exponentially many neurons in a shallow network but only polynomially many in a deep one. For instance, the function $x \mapsto \prod_{i=1}^d x_i$ can be computed by an $O(d)$-depth, $O(1)$-width network using the identity $ab = \frac{1}{4}[(a+b)^2 - (a-b)^2]$ recursively, but requires $\Omega(2^d)$ neurons in a single hidden layer (Telgarsky, 2016). The UAT is about width; the power of depth is a separate story.

2. **Training is not guaranteed.** The UAT guarantees expressive capacity, not that gradient descent will find the right weights. These are very different statements.

:::{prf:remark} UAT for ReLU
:label: rmk-uat-relu

For ReLU activations $\sigma(t) = \max(0,t)$, the UAT still holds. One way to see this: any ReLU network with $N$ hidden units computes a **piecewise linear function**, and piecewise linear functions are dense in $C(K)$ for any compact $K \subset \mathbb{R}^n$. More precisely, Lu et al. (2017) showed that depth-$L$ ReLU networks of width $\lceil n/L \rceil + 1$ suffice.

Interestingly, our own work ([Any Deep ReLU Network is Shallow](https://arxiv.org/pdf/2306.11827), ECAI 2025) shows that any depth-$L$ ReLU network can be rewritten as a depth-3 network if one allows weights in the extended reals $\overline{\mathbb{R}}$. This has consequences for interpretability, since depth and width become interchangeable given sufficient representational freedom.
:::

### No Free Lunch

The Universal Approximation Theorem might tempt one into thinking that neural networks are the final answer to all learning problems. The No Free Lunch Theorem (Wolpert & Macready, 1997) is the necessary sobering counterpoint: **there is no universally best learning algorithm**.

:::{prf:theorem} No Free Lunch Theorem
:label: thm-nfl

Let $\mathcal{X}$ be a finite input space and $\mathcal{Y} = \{0, 1\}$. Fix a training set size $m < |\mathcal{X}|$. For any two learning algorithms $\mathcal{A}$ and $\mathcal{B}$, when the target function $f: \mathcal{X} \to \mathcal{Y}$ is drawn uniformly at random from all $2^{|\mathcal{X}|}$ possible binary functions:

$$\mathbb{E}_f\bigl[\operatorname{err}_{\mathrm{off}}(\mathcal{A}, f)\bigr] = \mathbb{E}_f\bigl[\operatorname{err}_{\mathrm{off}}(\mathcal{B}, f)\bigr]$$

where $\operatorname{err}_{\mathrm{off}}(\mathcal{A}, f)$ is the off-training-set error of algorithm $\mathcal{A}$ on target $f$.
:::

:::{prf:proof}
:class: dropdown
:enumerated: false

Let the training set be $S = \{(x_i, f(x_i))\}_{i=1}^m$ for a fixed sequence of inputs $x_1, \ldots, x_m \in \mathcal{X}$, and let $T = \mathcal{X} \setminus \{x_1, \ldots, x_m\}$ be the test inputs. Any learning algorithm $\mathcal{A}$ takes $S$ as input and produces a hypothesis $h_\mathcal{A}^S: \mathcal{X} \to \{0,1\}$.

The off-training-set error is $\operatorname{err}_{\mathrm{off}}(\mathcal{A}, f, S) = \frac{1}{|T|} \sum_{x \in T} \mathbf{1}[h_\mathcal{A}^S(x) \neq f(x)]$.

Now we sum over all $2^{|\mathcal{X}|}$ target functions $f$. For any fixed $x \in T$ and any fixed hypothesis value $h_\mathcal{A}^S(x) \in \{0,1\}$, exactly half of all target functions satisfy $f(x) = h_\mathcal{A}^S(x)$ and half do not — because the value of $f$ on $T$ is completely free to vary, independently of its values on the training inputs. Therefore:

$$\sum_{f} \mathbf{1}[h_\mathcal{A}^S(x) \neq f(x)] = 2^{|\mathcal{X}|-1}$$

regardless of what $\mathcal{A}$ outputs. Summing over all $x \in T$ and all $f$, the total error is identical for every algorithm $\mathcal{A}$. $\square$
:::

The No Free Lunch Theorem is frequently misunderstood. It does **not** say that all algorithms are equally good in practice. It says that any advantage an algorithm has on one class of problems is precisely cancelled by its disadvantage on complementary classes — when we average over the uniform distribution over all problems.

The lesson is that **inductive bias is unavoidable and essential**. Every learning algorithm encodes prior assumptions about the structure of the target function. The extraordinary success of deep learning reflects a match between its inductive biases — locality, compositionality, smoothness, hierarchical feature extraction — and the structure of the data distributions we care about in the real world. There is no getting around this.

### The Neural Tangent Kernel

As neural networks grow very wide, their training dynamics simplify dramatically. This was the surprising discovery of Jacot, Gabriel and Hongler (NeurIPS 2018): in the infinite-width limit, training a neural network with gradient descent is equivalent to kernel regression with a fixed kernel called the **Neural Tangent Kernel**.

:::{prf:definition} Neural Tangent Kernel
:label: def-ntk

Let $f_\theta: \mathbb{R}^{d_0} \to \mathbb{R}$ be a neural network with parameters $\theta$. The **Neural Tangent Kernel (NTK)** is

$$\Theta_\theta(x, x') \coloneqq \bigl\langle \nabla_\theta f_\theta(x),\, \nabla_\theta f_\theta(x') \bigr\rangle_{\mathbb{R}^{|\theta|}}$$

the inner product of the parameter-space gradients at two inputs $x$ and $x'$.
:::

:::{prf:theorem} NTK Limit (Jacot et al., 2018)
:label: thm-ntk

Let $f_\theta^{(n)}$ be a fully-connected network of width $n$ with parameters initialised i.i.d. with appropriate $1/\sqrt{n}$ scaling. As $n \to \infty$:

1. The NTK $\Theta_\theta^{(n)}$ converges in probability to a deterministic limit $\Theta^\infty$ that depends only on the depth and activation, not on the random initialisation.
2. During gradient flow training, $\Theta_\theta^{(n)}(x, x')$ remains approximately constant (stays close to $\Theta^\infty$) for all time.
3. The training dynamics are therefore equivalent to **kernel gradient descent** with kernel $\Theta^\infty$.
:::

The NTK regime corresponds to **lazy training**: the parameters barely move from initialisation, and the network behaves like a linear model in the tangent space of the function space. This makes the loss landscape approximately convex near initialisation and allows precise convergence guarantees.

However, the NTK regime may not capture the most interesting behaviour. **Feature learning** — the process by which intermediate representations adapt to the data — happens precisely when networks operate *outside* the NTK regime. The practical success of deep learning likely comes from this feature-learning regime, where the network actively reshapes its internal representations. Understanding the interplay between these two regimes is an active area of research.

### Double Descent

Classical statistical wisdom holds that model complexity and test error follow a U-shaped curve: too simple and you underfit, too complex and you overfit. Neural networks famously violate this.

:::{prf:definition} Double Descent Phenomenon
:label: def-double-descent

Let $\mathcal{M}_n$ denote a family of models of increasing complexity $n$ (e.g. networks of increasing width). The **double descent** phenomenon refers to the observation that the test error $R(n)$ exhibits a qualitatively two-phase behaviour as $n$ increases:

1. **Classical (underparameterised) regime**: $R(n)$ first decreases then increases, following the classical bias–variance tradeoff.
2. **Interpolation threshold** ($n \approx n^*$): $R(n)$ spikes as the model just barely interpolates the training data — the model is "memorising" individual points.
3. **Modern (overparameterised) regime** ($n > n^*$): $R(n)$ decreases again, often falling below the classical minimum, even as the model perfectly fits the training data.
:::

The mathematical explanation for the third regime is subtle and still being developed. The key insight is that when a model has many more parameters than data points, gradient descent tends to find the **minimum-norm interpolant** — the simplest solution (in the parameter-space metric) that exactly fits the training data. In the linear case, this follows from the classical result that gradient descent initialised at zero converges to the minimum $\ell_2$-norm solution of an underdetermined linear system. In the NTK regime, the analogous result holds with respect to the RKHS norm of the Neural Tangent Kernel. For sufficiently expressive models with appropriate implicit regularisation, this minimum-norm interpolant generalises well.

This connects to several deep results: the **implicit bias of gradient descent** (which preferentially finds low-norm solutions in linear and nearly-linear models), the theory of **benign overfitting** (Bartlett et al., 2020), and the **NTK analysis** which shows that overparameterised networks trained with gradient descent converge to minimum-RKHS-norm solutions with respect to the NTK.

Double descent was documented empirically by Belkin et al. (2019) and has since been observed across models — from decision trees to transformers — and is widely considered a defining feature of the modern overparameterised learning paradigm.

---

## Learning

We have defined what neural networks are and stated fundamental results about their expressive power and limits. The natural next question is: how do we find good parameters?

This is the subject of **learning theory** and **optimisation**, which we will treat in subsequent sections. For now, the key idea is the following. We define a **loss function** $\ell: \mathcal{Y} \times \mathcal{Y} \to [0, \infty)$ measuring the discrepancy between network output $f_\theta(x_i)$ and true target $y_i$, and minimise the **empirical risk**:

$$\hat{R}(\theta) = \frac{1}{n} \sum_{i=1}^n \ell(f_\theta(x_i), y_i).$$

This is done via **stochastic gradient descent (SGD)** and its many variants, updating parameters iteratively:

$$\theta_{t+1} = \theta_t - \eta \, \nabla_\theta \ell\!\left(f_{\theta_t}(x_{i_t}), y_{i_t}\right)$$

for a learning rate $\eta > 0$ and a randomly sampled data point $(x_{i_t}, y_{i_t})$.

Why does SGD work despite the non-convexity of the loss landscape? This is one of the central open problems in deep learning theory. The combination of overparameterisation, implicit regularisation, and the geometry of the loss landscape conspires — in ways we do not fully understand — to make gradient descent find solutions that generalise remarkably well.
