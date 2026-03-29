# Natural Language Processing

Natural Language Processing (NLP) is the study of how machines can understand, generate, and reason about human language. It is arguably the area of AI where progress has been most dramatic — and most surprising. The discovery that scaling language models leads to emergent capabilities (reasoning, translation, coding, mathematics) has reshaped the entire field of AI.

At its core, NLP asks: how do we represent language mathematically, and how do we build models that capture its structure? The answer has converged, over the last decade, on a single architecture: the **Transformer**.

---

## Language as Sequences

Language is sequential: a sentence is an ordered sequence of tokens (words, subwords, or characters). The fundamental modelling choice is how to represent these tokens as mathematical objects that neural networks can process.

:::{prf:definition} Vocabulary and Tokenisation
:label: def-tokenisation

A **vocabulary** $\mathcal{V} = \{v_1, \ldots, v_{|\mathcal{V}|}\}$ is a finite set of tokens. A **tokeniser** is a function $\text{tok}: \Sigma^* \to \mathcal{V}^*$ that maps a string (over some alphabet $\Sigma$) to a sequence of tokens.

Common tokenisation schemes:
- **Word-level**: Each word is a token. Vocabulary is large ($|\mathcal{V}| \sim 100{,}000$); rare words are out-of-vocabulary.
- **Subword** (BPE, WordPiece, SentencePiece): Tokens are frequent substrings, learned from data. Vocabulary is moderate ($|\mathcal{V}| \sim 30{,}000\text{--}100{,}000$); any string can be encoded.
- **Character/byte-level**: Each character or byte is a token. Vocabulary is tiny ($|\mathcal{V}| \leq 256$); sequences are long.
:::

Modern language models use **subword tokenisation**, which balances vocabulary size against sequence length. The key insight of BPE (Byte Pair Encoding, Sennrich et al., 2016) is to iteratively merge the most frequent adjacent token pairs, building a vocabulary that captures common morphemes and words while remaining open to arbitrary text.

---

## Embeddings

Once text is tokenised, each token must be represented as a vector that the network can process.

:::{prf:definition} Token Embedding
:label: def-embedding

An **embedding** is a function $E: \mathcal{V} \to \mathbb{R}^d$ mapping each token to a $d$-dimensional vector. In practice, $E$ is parameterised as a learnable matrix $W_E \in \mathbb{R}^{|\mathcal{V}| \times d}$, and the embedding of token $v_i$ is the $i$-th row of $W_E$.
:::

The embedding dimension $d$ is a hyperparameter (typically $d \in \{256, 512, 768, 1024, \ldots\}$). The embeddings are learned during training, and after training, semantically similar tokens tend to have similar embedding vectors — a property known as the **distributional hypothesis**: "you shall know a word by the company it keeps" (Firth, 1957).

Early work on word embeddings (word2vec, Mikolov et al., 2013; GloVe, Pennington et al., 2014) showed that the learned embedding space has remarkable algebraic structure: $E(\text{king}) - E(\text{man}) + E(\text{woman}) \approx E(\text{queen})$. This linear structure in embedding space encodes semantic relationships — a precursor to the richer representations learned by modern Transformers.

---

## The Attention Mechanism

The Transformer architecture is built on a single primitive: **attention**. It is a mechanism for computing context-dependent representations by allowing each position in a sequence to "attend" to all other positions.

:::{prf:definition} Scaled Dot-Product Attention
:label: def-attention

Given **queries** $Q \in \mathbb{R}^{n \times d_k}$, **keys** $K \in \mathbb{R}^{n \times d_k}$, and **values** $V \in \mathbb{R}^{n \times d_v}$, the **scaled dot-product attention** is:

$$\text{Attention}(Q, K, V) = \operatorname{Softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

where $\operatorname{Softmax}$ is applied row-wise: each row of the attention weight matrix $A = \operatorname{Softmax}(QK^\top / \sqrt{d_k})$ is a probability distribution over positions.
:::

The intuition: each query "asks a question," each key "advertises what it contains," and the attention weights determine how much each value contributes to the output. The $\sqrt{d_k}$ scaling prevents the dot products from growing too large in high dimensions (which would push the softmax into saturation).

:::{prf:definition} Multi-Head Attention
:label: def-multi-head

**Multi-head attention** runs $h$ attention operations in parallel with different learned projections:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O$$

where $\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$ and $W_i^Q \in \mathbb{R}^{d \times d_k}$, $W_i^K \in \mathbb{R}^{d \times d_k}$, $W_i^V \in \mathbb{R}^{d \times d_v}$, $W^O \in \mathbb{R}^{hd_v \times d}$ are learnable projection matrices.
:::

Multiple heads allow the model to attend to different aspects of the input simultaneously: one head might capture syntactic relationships, another semantic similarity, another positional proximity. The heads are combined linearly and the model learns which aspects are useful for each task.

---

## The Transformer

The **Transformer** (Vaswani et al., 2017, "Attention Is All You Need") assembles attention and feedforward layers into a complete architecture. It has largely replaced RNNs and CNNs for sequence modelling.

:::{prf:definition} Transformer Block
:label: def-transformer-block

A **Transformer block** applies the following operations to an input sequence $X \in \mathbb{R}^{n \times d}$:

1. **Multi-head self-attention** with residual connection and layer normalisation:
$$X' = \text{LayerNorm}(X + \text{MultiHead}(X, X, X))$$

2. **Position-wise feedforward network** with residual connection and layer normalisation:
$$X'' = \text{LayerNorm}(X' + \text{FFN}(X'))$$

where $\text{FFN}(x) = W_2 \, \sigma(W_1 x + b_1) + b_2$ is a two-layer feedforward network applied independently to each position.

A Transformer with $L$ blocks stacks $L$ such operations sequentially.
:::

Key design choices:
- **Self-attention** ($Q = K = V = X$): each position attends to all positions in the same sequence.
- **Causal masking** (for autoregressive models): position $i$ can only attend to positions $j \leq i$, enforced by setting $A_{ij} = -\infty$ for $j > i$ before the softmax.
- **Positional encoding**: Since attention is permutation-equivariant, position information must be added explicitly. Sinusoidal encodings ($\text{PE}(pos, 2i) = \sin(pos / 10000^{2i/d})$) or learned positional embeddings are common. Modern architectures use **Rotary Position Embeddings (RoPE)** which encode relative positions directly in the attention computation.

---

## Language Modelling

The dominant paradigm in modern NLP is **language modelling**: training a model to predict the next token in a sequence.

:::{prf:definition} Autoregressive Language Model
:label: def-language-model

An **autoregressive language model** defines a probability distribution over sequences $x = (x_1, \ldots, x_T)$ by factoring via the chain rule:

$$p_\theta(x_1, \ldots, x_T) = \prod_{t=1}^T p_\theta(x_t \mid x_1, \ldots, x_{t-1}).$$

Each conditional $p_\theta(x_t \mid x_{<t})$ is computed by a Transformer with causal masking, followed by a linear projection to the vocabulary and a softmax:

$$p_\theta(x_t = v \mid x_{<t}) = \operatorname{Softmax}(W_{\text{head}} \cdot h_t)_v$$

where $h_t \in \mathbb{R}^d$ is the Transformer's hidden representation at position $t$.
:::

Training maximises the log-likelihood of the training corpus — equivalently, minimises the **cross-entropy loss**:

$$\mathcal{L}(\theta) = -\frac{1}{T}\sum_{t=1}^T \log p_\theta(x_t \mid x_{<t}).$$

This single objective — predict the next token — turns out to be extraordinarily powerful. A model that predicts text well must implicitly capture syntax, semantics, world knowledge, reasoning patterns, and much more. The "unreasonable effectiveness" of language modelling is one of the central empirical discoveries of modern AI.

---

## Scaling Laws

A remarkable empirical finding is that language model performance (measured by cross-entropy loss) follows predictable **power laws** in model size, dataset size, and compute.

:::{prf:definition} Neural Scaling Laws (Kaplan et al., 2020)
:label: def-scaling-laws

For Transformer language models, the test loss $L$ scales as:

$$L(N) \approx \left(\frac{N_c}{N}\right)^{\alpha_N}, \quad L(D) \approx \left(\frac{D_c}{D}\right)^{\alpha_D}, \quad L(C) \approx \left(\frac{C_c}{C}\right)^{\alpha_C}$$

where $N$ is the number of parameters, $D$ is the dataset size (in tokens), and $C$ is the compute budget (in FLOPs). The exponents $\alpha_N \approx 0.076$, $\alpha_D \approx 0.095$, $\alpha_C \approx 0.050$ are approximately universal across model families.
:::

Scaling laws have profound implications for the practice of AI research:

1. **Predictability**: We can extrapolate the performance of a large model from smaller experiments. This makes large-scale training economically rational — you know approximately what you will get before you spend the compute.
2. **Compute-optimal training** (Hoffmann et al., 2022, "Chinchilla"): For a fixed compute budget, there is an optimal allocation between model size and data. The Chinchilla scaling law suggests that most large models are **undertrained** — one should scale data proportionally with parameters.
3. **Emergent capabilities**: Some capabilities (multi-step reasoning, chain-of-thought, in-context learning) appear only above certain scale thresholds, suggesting phase transitions in the loss landscape.

The theoretical explanation for these power laws remains an open problem. Why should loss follow a power law? One hypothesis connects to the **intrinsic dimensionality** of natural language: if the data lies on a manifold of dimension $d_{\text{int}}$, then the approximation error scales as $N^{-2/d_{\text{int}}}$, which is a power law.
