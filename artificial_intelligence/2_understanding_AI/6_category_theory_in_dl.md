# Category Theory in Deep Learning

I am not a category theorist. My knowledge is basic, but I have read enough category theory and worked closely enough with category theorists — including my good friend Bruno Gavranovic and Dom Verity at Symbolica — to appreciate the value of categorical thinking for deep learning. This chapter collects what I have found most useful.

Category theory is sometimes described as "the mathematics of mathematics" — it studies the structure of mathematical structures themselves. For deep learning, its value is threefold:

1. **Unification**: Different definitions of neural networks (parametrised functions, conditional distributions, dynamical systems) are morphisms in different categories. Category theory organises these perspectives and makes the relationships precise.
2. **Compositionality**: Neural networks are built by composing modules. Categories are built to reason about composition. The match is natural.
3. **Prescription**: Specifying the categorical structure of a problem can determine the correct architecture, just as specifying the symmetry group determines equivariant layers in Geometric Deep Learning.

For rigorous study, I recommend Emily Riehl's *Category Theory in Context*, Bartosz Milewski's blog and YouTube lectures (especially for programmers), and Bruno Gavranovic's thesis and papers for the AI applications.

---

## Categories, Functors, Natural Transformations

We recall the basic definitions; they were introduced briefly in the [Neural Networks](1_neural_networks.md) chapter.

:::{prf:definition} Category
:label: def-category-dl

A **category** $\mathcal{C}$ consists of:
- A collection of **objects** $\text{ob}(\mathcal{C})$.
- For each pair of objects $A, B$, a collection of **morphisms** $\text{Hom}_\mathcal{C}(A, B)$.
- A **composition** operation: if $f: A \to B$ and $g: B \to C$, then $g \circ f: A \to C$.
- An **identity** morphism $\text{id}_A: A \to A$ for each object.

Composition is associative and identities are neutral: $f \circ \text{id}_A = f = \text{id}_B \circ f$.
:::

The categories relevant to deep learning include:
- $\textbf{Vect}_\mathbb{R}$: Vector spaces and linear maps — the linear algebra layer.
- $\textbf{Smooth}$: Smooth manifolds and smooth maps — the continuous geometry.
- $\textbf{Meas}$: Measurable spaces and measurable functions — the probabilistic setting.
- $\textbf{Markov}$: Probability spaces and Markov kernels — stochastic computation.

:::{prf:definition} Functor
:label: def-functor-dl

A **functor** $F: \mathcal{C} \to \mathcal{D}$ between categories maps objects to objects and morphisms to morphisms, preserving composition and identities:

$$F(g \circ f) = F(g) \circ F(f), \quad F(\text{id}_A) = \text{id}_{F(A)}.$$
:::

Functors are "structure-preserving maps between categories." A key example: the **forgetful functor** $U: \textbf{Vect}_\mathbb{R} \to \textbf{Set}$ maps each vector space to its underlying set, "forgetting" the linear structure. Going the other way, the **free functor** $F: \textbf{Set} \to \textbf{Vect}_\mathbb{R}$ maps a set to the free vector space on it.

:::{prf:definition} Natural Transformation
:label: def-natural-transformation

Given functors $F, G: \mathcal{C} \to \mathcal{D}$, a **natural transformation** $\eta: F \Rightarrow G$ is a family of morphisms $\eta_A: F(A) \to G(A)$ for each object $A \in \mathcal{C}$, such that for every morphism $f: A \to B$:

$$\eta_B \circ F(f) = G(f) \circ \eta_A.$$

This is the **naturality condition**: the square diagram commutes.
:::

Natural transformations are "maps between functors that respect the structure." They are how we formalise the idea that two constructions are "the same in a structured way."

---

## The Para Construction

The most important categorical construction for deep learning is $\textbf{Para}$, which we introduced in the neural networks chapter. Let us develop it further.

:::{prf:definition} The Category $\textbf{Para}(\mathcal{C})$
:label: def-para-full

Given a symmetric monoidal category $(\mathcal{C}, \otimes, I)$, the category $\textbf{Para}(\mathcal{C})$ has:

- **Objects**: Same as $\mathcal{C}$.
- **Morphisms** from $A$ to $B$: Pairs $(P, f)$ where $P \in \mathcal{C}$ is a **parameter object** and $f: P \otimes A \to B$ is a morphism in $\mathcal{C}$.
- **Composition**: $(P_2, g) \circ (P_1, f) = (P_1 \otimes P_2,\; g \circ (\text{id}_{P_2} \otimes f))$ — the parameter objects tensor together.
- **Identity**: $(I, \text{id}_A)$ where $I$ is the monoidal unit.
:::

A neural network layer is a morphism in $\textbf{Para}(\textbf{Smooth})$: the parameter object is $\mathbb{R}^{d_\ell \times d_{\ell-1}} \times \mathbb{R}^{d_\ell}$ (weights and biases), and the morphism is the affine-plus-activation map. An $L$-layer network is a composition of $L$ such morphisms, with total parameter space $P_1 \otimes \cdots \otimes P_L$.

This makes **modularity** categorical: modules compose, parameter spaces tensor, and architectural choices are choices of morphism. Replacing one module with another (say, swapping a convolutional layer for an attention layer) is replacing one morphism in $\textbf{Para}$ with another, provided the types (input/output dimensions) match.

---

## Lenses and Backpropagation

Backpropagation — the workhorse algorithm of deep learning — has a categorical formulation in terms of **lenses** (or **optics**).

:::{prf:definition} Lens
:label: def-lens

A **lens** from $(A, A')$ to $(B, B')$ is a pair of maps:

$$\text{get}: A \to B, \quad \text{put}: A \times B' \to A'$$

The **get** map computes the forward pass (from input to output), and the **put** map computes the backward pass (from input and output gradient to input gradient).
:::

A neural network layer $f_\theta: A \to B$ defines a lens where:
- $\text{get}(a) = f_\theta(a)$ is the forward pass.
- $\text{put}(a, \delta b) = \delta b \cdot \nabla_a f_\theta(a)$ is the backward pass (chain rule application).

Composition of lenses corresponds to composition of forward and backward passes — which is exactly the backpropagation algorithm. The category of lenses, $\textbf{Lens}$, makes this precise.

:::{prf:remark}
:label: rmk-optics

The lens framework generalises to **optics** (Riley, 2018), which handle more complex information flow patterns: mixed optics for different forward/backward types, dependent optics for attention mechanisms, and Tambara modules for equivariant networks. The categorical perspective on backpropagation is developed in detail by Fong, Spivak & Tuyéras (2019, "Backprop as Functor") and Cruttwell et al. (2022, "Categorical Foundations of Gradient-Based Learning").
:::

---

## The Learner Category

Training a neural network involves three things: a model, a loss function, and an update rule. All three can be unified categorically.

:::{prf:definition} Learner (Fong, Spivak & Tuyéras, 2019)
:label: def-learner

A **learner** from $A$ to $B$ is a tuple $(P, I, U, r)$ where:
- $P$ is a parameter space,
- $I: P \times A \to B$ is the **implementation** (forward pass),
- $U: P \times A \times B \to P$ is the **update rule** (parameter update given input and feedback),
- $r: B \times B \to B$ is a **request** function (gradient of loss with respect to output).

Learners compose: chaining two learners corresponds to backpropagating through two layers.
:::

The learner category makes the entire training pipeline — forward pass, loss computation, gradient computation, parameter update — into a single categorical object. The compositionality of learners is the compositionality of backpropagation.

---

## Yoneda's Lemma

Emily Riehl once said that she returned to Yoneda's Lemma repeatedly throughout her career and found new depth each time. I am a much more basic user of category theory than her, but I have come to appreciate the lemma as a statement about **meaning**.

:::{prf:theorem} Yoneda's Lemma
:label: thm-yoneda

For any locally small category $\mathcal{C}$, any object $A \in \mathcal{C}$, and any functor $F: \mathcal{C}^{\text{op}} \to \textbf{Set}$, there is a natural bijection:

$$\text{Nat}(\text{Hom}(-, A), F) \cong F(A).$$

Natural transformations from the representable functor $\text{Hom}(-, A)$ to $F$ are in bijection with elements of $F(A)$.
:::

The deep content: **an object is completely determined by its relationships to all other objects**. The representable functor $\text{Hom}(-, A)$ encodes "all the ways things can map into $A$," and Yoneda says this is enough to recover $A$ up to isomorphism.

For deep learning, Yoneda's Lemma has several resonances:

1. **Embeddings**: A word embedding $E: \mathcal{V} \to \mathbb{R}^d$ represents each word by its "relationships" (co-occurrence patterns, contextual neighbours). Yoneda says: if the embedding preserves all relationships, it preserves all information. This is the mathematical justification for the distributional hypothesis.

2. **Representations**: A neural network's internal representation of an input $x$ — the hidden state $h(x)$ — is determined by how $x$ interacts with all other inputs under the model. This is exactly the Yoneda perspective: the representation is the collection of morphisms from $x$.

3. **Interpretability**: Understanding what a neural network has learned is, in a Yoneda sense, understanding the relationships between its internal representations. Mechanistic interpretability (probing, circuits) is an attempt to decompose the representable functor into understandable components.

---

## Categorical Perspectives on Architecture

Different neural network architectures correspond to morphisms in different categories. This perspective prescribes architectures:

| Architecture | Category | Morphisms | Key Structure |
|---|---|---|---|
| Feedforward NN | $\textbf{Para}(\textbf{Smooth})$ | Parametrised smooth maps | Composition |
| Bayesian NN | $\textbf{Para}(\textbf{Markov})$ | Parametrised Markov kernels | Stochastic composition |
| GCN | $\textbf{Para}(\textbf{CoKl}(\mathcal{N}))$ | Parametric coKleisli morphisms | Neighbourhood comonad $\mathcal{N}$ |
| Equivariant NN | $\textbf{Para}(\textbf{Rep}_G)$ | $G$-equivariant parametric maps | Group representation |
| RNN | $\textbf{Para}(\textbf{Smooth})$ (with state) | Stateful parametric maps | Monoidal state |

Each row tells you: if your data has a particular structure (graph → comonad, symmetry → group representation), the categorical framework determines the space of valid operations.

The vision — which is still largely aspirational — is that category theory could serve as a **type system for deep learning**: specifying the categorical type of a problem automatically generates the correct architecture, in the same way that specifying a type signature in Haskell constrains the space of valid programs.
