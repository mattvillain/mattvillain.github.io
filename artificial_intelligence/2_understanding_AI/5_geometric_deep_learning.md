# Geometric Deep Learning

Why do CNNs work so well on images? Why do GNNs work on molecules? Why does the Transformer dominate sequence modelling? The answer, in each case, is **symmetry**: each architecture encodes a specific symmetry of the data domain, and this symmetry acts as a powerful inductive bias that constrains the hypothesis class to functions that are consistent with the problem's geometry.

**Geometric Deep Learning** (Bronstein et al., 2021) is the programme of understanding neural network architectures through the lens of symmetry and invariance. It provides a unifying framework — sometimes called the **5G framework** (Grids, Groups, Graphs, Geodesics, Gauges) — that derives known architectures as special cases and suggests new ones.

I got into this area because I thought category theory could help — and it did. Our work on [Graph Convolutional Neural Networks as Parametric CoKleisli Morphisms](https://arxiv.org/abs/2212.00542) (SYCO-10, 2022) with Bruno Gavranovic was an attempt to formalise the weight-sharing structure of GCNs in categorical terms. This chapter gives the mathematical foundations.

---

## Symmetry and Groups

The mathematical language for symmetry is **group theory**. A group captures the set of transformations that leave some structure unchanged.

:::{prf:definition} Group
:label: def-group

A **group** $(G, \cdot)$ is a set $G$ equipped with a binary operation $\cdot: G \times G \to G$ satisfying:
1. **Associativity**: $(g_1 \cdot g_2) \cdot g_3 = g_1 \cdot (g_2 \cdot g_3)$.
2. **Identity**: There exists $e \in G$ such that $g \cdot e = e \cdot g = g$ for all $g$.
3. **Inverses**: For each $g \in G$, there exists $g^{-1} \in G$ with $g \cdot g^{-1} = e$.
:::

**Examples relevant to deep learning:**
- $(\mathbb{Z}^2, +)$: Translations of a 2D grid — the symmetry group of CNNs.
- $S_n$: Permutations of $n$ elements — the symmetry group of sets and graphs.
- $SO(3)$: 3D rotations — the symmetry group of molecular and physical systems.
- $SE(3) = SO(3) \ltimes \mathbb{R}^3$: Rigid motions (rotations + translations) — the symmetry group of point clouds and proteins.

:::{prf:definition} Group Action and Representation
:label: def-group-action

A **group action** of $G$ on a set $\mathcal{X}$ is a map $\rho: G \times \mathcal{X} \to \mathcal{X}$ satisfying $\rho(e, x) = x$ and $\rho(g_1, \rho(g_2, x)) = \rho(g_1 g_2, x)$.

A **representation** is a group action on a vector space: a homomorphism $\rho: G \to GL(V)$ mapping group elements to invertible linear maps. We write $\rho_g(x) = \rho(g) \cdot x$.
:::

---

## Equivariance and Invariance

The central concepts of Geometric Deep Learning are **equivariance** and **invariance**: constraints on how a function transforms when its input is transformed.

:::{prf:definition} Equivariance and Invariance
:label: def-equivariance

Let $G$ act on input space $\mathcal{X}$ via representation $\rho_{\text{in}}$ and on output space $\mathcal{Y}$ via $\rho_{\text{out}}$. A function $f: \mathcal{X} \to \mathcal{Y}$ is:

- **$G$-equivariant** if $f(\rho_{\text{in}}(g) \cdot x) = \rho_{\text{out}}(g) \cdot f(x)$ for all $g \in G$, $x \in \mathcal{X}$.
- **$G$-invariant** if $f(\rho_{\text{in}}(g) \cdot x) = f(x)$ for all $g \in G$, $x \in \mathcal{X}$ (equivariance with trivial output representation).
:::

Equivariance says: transforming the input and then computing is the same as computing and then transforming the output. Invariance is the special case where the output doesn't change at all.

**Examples:**
- A CNN layer is **translation-equivariant**: translating the input image translates the output feature map by the same amount.
- A graph neural network layer is **permutation-equivariant**: permuting the nodes of the input graph permutes the output node features in the same way.
- A classifier should be **translation-invariant**: the label "cat" should not depend on where the cat appears.

:::{prf:theorem} Equivariance Constrains the Weight Space
:label: thm-equivariance-constraint

A linear map $f: \mathbb{R}^{d_{\text{in}}} \to \mathbb{R}^{d_{\text{out}}}$ is $G$-equivariant (with respect to representations $\rho_{\text{in}}$ and $\rho_{\text{out}}$) if and only if its weight matrix $W$ satisfies:

$$W \rho_{\text{in}}(g) = \rho_{\text{out}}(g) W \quad \text{for all } g \in G.$$

The space of equivariant linear maps is a subspace of all linear maps, and its dimension equals the number of free parameters in the equivariant architecture.
:::

:::{prf:proof}
:class: dropdown
:enumerated: false

The equivariance condition $f(\rho_{\text{in}}(g) x) = \rho_{\text{out}}(g) f(x)$ for $f(x) = Wx$ becomes $W\rho_{\text{in}}(g)x = \rho_{\text{out}}(g)Wx$ for all $x$. Since this must hold for all $x$, we get $W\rho_{\text{in}}(g) = \rho_{\text{out}}(g)W$. This is a system of linear constraints on the entries of $W$; the solution space is a linear subspace. $\square$
:::

This theorem is the theoretical engine of Geometric Deep Learning: specifying the symmetry group $G$ and its representations **determines** the space of allowed operations, which in turn determines the architecture.

---

## CNNs as Equivariant Networks

Convolution is not an arbitrary architectural choice — it is the **unique** translation-equivariant linear operation (up to the choice of kernel).

:::{prf:theorem} Convolution Characterises Translation Equivariance
:label: thm-conv-equiv

A linear map $f: L^2(\mathbb{R}^n) \to L^2(\mathbb{R}^n)$ is equivariant to translations $T_s: x(t) \mapsto x(t - s)$ if and only if $f$ is a **convolution**: $f(x) = k * x$ for some kernel $k$.
:::

This result explains why CNNs work: if you believe the relevant patterns in images can appear anywhere (translation symmetry), then convolution is the uniquely correct linear operation. The only design freedom is the kernel $k$, which is learned from data.

---

## Graph Neural Networks

For data defined on graphs (social networks, molecules, citation networks), the relevant symmetry group is the **permutation group** $S_n$: relabelling the nodes should not change the computation.

:::{prf:definition} Message Passing Neural Network
:label: def-mpnn

A **message passing neural network** (MPNN) (Gilmer et al., 2017) updates node representations through iterative neighbourhood aggregation:

$$h_v^{(\ell+1)} = \phi^{(\ell)}\!\left(h_v^{(\ell)},\, \bigoplus_{u \in \mathcal{N}(v)} \psi^{(\ell)}(h_v^{(\ell)}, h_u^{(\ell)}, e_{vu})\right)$$

where $h_v^{(\ell)} \in \mathbb{R}^d$ is the representation of node $v$ at layer $\ell$, $\mathcal{N}(v)$ is the set of neighbours of $v$, $\psi^{(\ell)}$ is a **message function**, $\bigoplus$ is a permutation-invariant **aggregation** (sum, mean, or max), and $\phi^{(\ell)}$ is an **update function**.
:::

The aggregation $\bigoplus$ must be permutation-invariant (the sum of a set doesn't depend on the order) — this is what makes MPNNs permutation-equivariant. The message and update functions are typically neural networks.

**Graph Convolutional Networks (GCNs)** (Kipf & Welling, 2017) are a specific instance with $\psi(h_v, h_u) = W h_u$ and $\bigoplus = \text{normalised sum}$:

$$h_v^{(\ell+1)} = \sigma\!\left(\sum_{u \in \mathcal{N}(v) \cup \{v\}} \frac{1}{\sqrt{d_v d_u}} W^{(\ell)} h_u^{(\ell)}\right).$$

:::{prf:remark}
:label: rmk-gcn-cokleisli

In our work with Bruno Gavranovic ([GCNs as Parametric CoKleisli Morphisms](https://arxiv.org/abs/2212.00542)), we showed that the weight-sharing structure of GCNs can be formalised categorically. The neighbourhood aggregation is a **coKleisli morphism** for a comonad that extracts local neighbourhoods from the graph, and the shared weights make this a **parametric** coKleisli morphism in $\textbf{Para}(\textbf{Smooth})$. This categorical perspective makes precise why GCNs are the "right" architecture for graphs — they are determined by the comonadic structure of local-to-global information flow.
:::

---

## The 5G Framework

Bronstein et al. (2021) proposed a unifying taxonomy that organises architectures by their symmetry domain:

| Domain | Symmetry Group | Architecture | Equivariance |
|---|---|---|---|
| **Grids** ($\mathbb{Z}^d$) | Translation $(\mathbb{Z}^d, +)$ | CNN | Translation equivariance |
| **Groups** ($G$) | Group $G$ | Group convolution | $G$-equivariance |
| **Graphs** | Permutation $S_n$ | MPNN / GCN | Node permutation equivariance |
| **Geodesics** (manifolds) | Isometry group | Gauge CNN | Local gauge equivariance |
| **Gauges** (fibre bundles) | Structure group | Gauge-equivariant NN | Gauge equivariance |

Each row recovers a known architecture class as the unique equivariant network for that domain and symmetry. The framework is prescriptive: given a new domain with a known symmetry, it tells you what architecture to build.

:::{prf:remark}
:label: rmk-gdl-limits

The Geometric Deep Learning programme has a fundamental limitation: it requires knowing the symmetry group in advance. For many real-world problems, the relevant symmetries are approximate, partial, or unknown. Learning symmetries from data — **symmetry discovery** — is an active research direction. Moreover, Transformers, which make minimal symmetry assumptions, often outperform specialised equivariant architectures when given sufficient data. The interplay between data-driven generality and symmetry-based efficiency remains one of the central tensions in architecture design.
:::
