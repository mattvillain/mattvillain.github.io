# Architecture and Modelling

The choice of neural network architecture determines what kinds of functions the model can represent efficiently, what inductive biases it encodes, and how information flows during both forward computation and gradient-based training. Architecture design is, in a sense, the engineering art of AI: choosing the right structural constraints to match the structure of the problem.

This chapter surveys the major architectural families and the principles behind them. We will see that each architecture encodes a specific assumption about the data — locality for CNNs, sequential dependence for RNNs, pairwise interactions for Transformers — and that understanding these assumptions is key to understanding when and why they work.

---

## Convolutional Neural Networks

**Convolutional Neural Networks (CNNs)** are the canonical architecture for data with spatial or grid-like structure: images, audio spectrograms, time series. Their key inductive bias is **translation equivariance**: the same pattern should be detected regardless of where it appears in the input.

:::{prf:definition} Convolutional Layer
:label: def-conv-layer

A **convolutional layer** applies a learnable filter (kernel) $K \in \mathbb{R}^{k \times k \times c_{\text{in}}}$ to an input feature map $X \in \mathbb{R}^{h \times w \times c_{\text{in}}}$ to produce an output feature map $Y \in \mathbb{R}^{h' \times w' \times c_{\text{out}}}$:

$$(Y)_{i,j,f} = \sigma\!\left(\sum_{a,b,c} K_{a,b,c}^{(f)} \cdot X_{i+a, j+b, c} + b_f\right)$$

where $f$ indexes the output channel, $(a, b)$ range over the kernel window, $c$ ranges over input channels, and $\sigma$ is a nonlinear activation.
:::

The critical property: the kernel $K$ is **shared** across all spatial positions $(i,j)$. This parameter sharing has two consequences:

1. **Translation equivariance**: If the input shifts by $(s_1, s_2)$, the output shifts by the same amount. A cat detector activates wherever the cat appears.
2. **Parameter efficiency**: A $3 \times 3$ kernel on $256$-channel inputs has $3 \times 3 \times 256 \times c_{\text{out}}$ parameters, regardless of the spatial dimensions $h \times w$. A fully-connected layer would have $h \cdot w \cdot 256 \cdot c_{\text{out}}$ parameters.

**Pooling** (max-pooling, average-pooling) provides **approximate translation invariance** by aggregating over spatial neighbourhoods, reducing resolution and making the representation progressively more abstract.

Classic CNN architectures — LeNet (1989), AlexNet (2012), VGG (2014), ResNet (2015) — established the template: stack convolutional layers with nonlinearities and pooling, progressively increasing the number of channels while decreasing spatial resolution, and attach a classifier head at the end.

---

## Recurrent Neural Networks

**Recurrent Neural Networks (RNNs)** process sequential data by maintaining a **hidden state** that is updated at each time step. They are the natural architecture for sequences: text, speech, time series.

:::{prf:definition} Recurrent Neural Network
:label: def-rnn

An **RNN** maps an input sequence $(x_1, \ldots, x_T)$ to a sequence of hidden states $(h_1, \ldots, h_T)$ via the recurrence:

$$h_t = \sigma(W_h h_{t-1} + W_x x_t + b)$$

where $W_h \in \mathbb{R}^{d_h \times d_h}$, $W_x \in \mathbb{R}^{d_h \times d_x}$ are shared weight matrices and $\sigma$ is an activation function.
:::

The same weights $(W_h, W_x)$ are applied at every time step — another form of parameter sharing, analogous to the spatial weight sharing in CNNs. The hidden state $h_t$ summarises the entire history $(x_1, \ldots, x_t)$ into a fixed-dimensional vector.

The fundamental problem with vanilla RNNs is the **vanishing/exploding gradient**: when backpropagating through $T$ time steps, the gradient involves the product $\prod_{t=1}^T \nabla_{h_{t-1}} h_t$, which tends to either vanish or explode exponentially with $T$. This makes it difficult to learn long-range dependencies.

:::{prf:definition} Long Short-Term Memory (LSTM)
:label: def-lstm

An **LSTM** (Hochreiter & Schmidhuber, 1997) addresses the vanishing gradient by introducing a **cell state** $c_t$ and three **gates** that control information flow:

- **Forget gate**: $f_t = \sigma(W_f [h_{t-1}, x_t] + b_f)$ — what to discard from cell state.
- **Input gate**: $i_t = \sigma(W_i [h_{t-1}, x_t] + b_i)$ — what new information to store.
- **Output gate**: $o_t = \sigma(W_o [h_{t-1}, x_t] + b_o)$ — what to output.

The cell state and hidden state are updated as:
$$c_t = f_t \odot c_t + i_t \odot \tanh(W_c [h_{t-1}, x_t] + b_c), \quad h_t = o_t \odot \tanh(c_t)$$

where $\odot$ denotes element-wise multiplication.
:::

The key insight: the cell state $c_t$ follows an **additive** update rule (gated addition, not matrix multiplication), which provides a highway for gradients to flow across many time steps without vanishing. The gates learn when to remember, forget, and output — they are themselves neural networks.

LSTMs dominated sequence modelling from 2014 to 2017. They have been largely replaced by Transformers, which avoid the sequential bottleneck entirely via attention.

---

## Residual Networks and Depth

**Residual networks (ResNets)** (He et al., 2015) introduced the **skip connection**: instead of learning a function $F(x)$, each layer learns a **residual** $F(x) - x$, and the output is $x + F(x)$.

:::{prf:definition} Residual Block
:label: def-residual

A **residual block** computes:

$$x_{\ell+1} = x_\ell + F_\ell(x_\ell)$$

where $F_\ell$ is a neural network (typically two convolutional layers with batch normalisation and ReLU). The skip connection adds the input directly to the output.
:::

Skip connections solve the **degradation problem**: very deep networks (50+ layers) trained without skip connections perform *worse* than shallower networks, even on the training set. This is not overfitting — it is an optimisation failure. Skip connections fix this by providing a gradient highway: even if $F_\ell$ produces small gradients, the identity branch ensures $\partial x_{\ell+1}/\partial x_\ell = I + \partial F_\ell/\partial x_\ell$, which is well-conditioned.

:::{prf:remark}
:label: rmk-resnet-ode

The residual update $x_{\ell+1} = x_\ell + F_\ell(x_\ell)$ is an **Euler discretisation** of the ODE $dx/dt = F(x, t)$ with step size $h = 1$. This observation led to **Neural ODEs** (Chen et al., 2018), which replace the discrete layers with a continuous dynamics solved by an ODE integrator. The depth of the network becomes a continuous variable — the integration time — and the number of function evaluations adapts to the complexity of the input. This perspective connects deep learning to dynamical systems theory and has applications in generative modelling and time-series analysis.
:::

---

## Normalisation

Training deep networks requires careful control of the distribution of activations across layers. **Normalisation** techniques stabilise training by ensuring that the inputs to each layer have controlled statistics.

:::{prf:definition} Batch Normalisation
:label: def-batch-norm

**Batch normalisation** (Ioffe & Szegedy, 2015) normalises activations across the batch dimension. For a mini-batch $\{x_1, \ldots, x_B\}$ at a given layer:

$$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}, \quad y_i = \gamma \hat{x}_i + \beta$$

where $\mu_B = \frac{1}{B}\sum_i x_i$, $\sigma_B^2 = \frac{1}{B}\sum_i (x_i - \mu_B)^2$ are batch statistics, $\epsilon$ is a small constant for numerical stability, and $\gamma, \beta$ are learnable scale and shift parameters.
:::

:::{prf:definition} Layer Normalisation
:label: def-layer-norm

**Layer normalisation** (Ba et al., 2016) normalises across the feature dimension instead of the batch dimension:

$$\hat{x} = \frac{x - \mu_x}{\sqrt{\sigma_x^2 + \epsilon}}, \quad y = \gamma \odot \hat{x} + \beta$$

where $\mu_x$ and $\sigma_x^2$ are computed over the features of a single input $x$.
:::

Batch normalisation was transformative for CNNs but is problematic for Transformers (it depends on batch size and is not well-defined for variable-length sequences). **Layer normalisation** is the standard for Transformers: it normalises each token's representation independently, making it batch-size agnostic and compatible with autoregressive generation.

---

## Initialisation

Before training begins, network parameters must be initialised. Poor initialisation can cause activations to explode or vanish through the layers, preventing training from starting at all.

:::{prf:definition} Xavier and He Initialisation
:label: def-initialisation

For a linear layer $y = Wx$ with $d_{\text{in}}$ inputs and $d_{\text{out}}$ outputs:

- **Xavier initialisation** (Glorot & Bengio, 2010): $W_{ij} \sim \mathcal{N}(0, 2/(d_{\text{in}} + d_{\text{out}}))$. Preserves variance for linear and tanh activations.
- **He initialisation** (He et al., 2015): $W_{ij} \sim \mathcal{N}(0, 2/d_{\text{in}})$. Designed for ReLU activations, which halve the variance (since $\mathbb{E}[\text{ReLU}(x)^2] = \frac{1}{2}\mathbb{E}[x^2]$ for symmetric $x$).
:::

The principle is **variance preservation**: if the input to each layer has unit variance, the output should also have approximately unit variance. This ensures that signals propagate through the network without amplification or attenuation — a necessary condition for gradients to flow in the backward pass.

Modern architectures often combine careful initialisation with normalisation layers. The two are complementary: initialisation sets the right scale at the start of training, and normalisation maintains it throughout.

---

## Architectural Principles

Looking across these families, several recurring principles emerge:

1. **Parameter sharing** encodes symmetry: CNNs share across space (translation equivariance), RNNs share across time (temporal homogeneity), and Transformers share across positions via the same attention mechanism. The [Geometric Deep Learning](5_geometric_deep_learning.md) chapter makes this connection to group theory precise.

2. **Residual connections** enable depth by providing gradient highways. They appear in ResNets, Transformers, and virtually every modern architecture.

3. **Normalisation** stabilises training by controlling activation statistics. The choice of normalisation (batch, layer, group, instance) depends on the architecture and the data modality.

4. **Attention** allows global interactions with adaptive, input-dependent connectivity. It has progressively replaced fixed, local connectivity (convolution) and sequential processing (recurrence) across domains.

The trend in architecture design is toward greater generality: Transformers, which make minimal structural assumptions, have proven effective across modalities (text, images, audio, biology, code). The trade-off is that generality requires more data and compute — specific inductive biases (like convolution for images) can be more sample-efficient when they match the data structure.
