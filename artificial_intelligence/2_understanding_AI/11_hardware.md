# Hardware

Everything discussed in this book — every gradient computation, every attention matrix, every Monte Carlo simulation — ultimately runs on physical hardware. The extraordinary progress in deep learning over the past decade is as much a story about hardware as about algorithms. Understanding the hardware landscape is essential for understanding why certain architectures succeed, why training costs what it does, and where the bottlenecks lie.

---

## The Computational Cost of Deep Learning

Training a neural network involves two operations at every layer: a **forward pass** (compute the output) and a **backward pass** (compute the gradient). Both are dominated by matrix multiplications.

:::{prf:definition} FLOPs for a Linear Layer
:label: def-flops-linear

A linear layer $y = Wx + b$ with $W \in \mathbb{R}^{m \times n}$ requires:
- **Forward pass**: $2mn$ FLOPs (one multiply-add per output element per input element).
- **Backward pass** (gradient w.r.t. input): $2mn$ FLOPs.
- **Backward pass** (gradient w.r.t. weights): $2mn$ FLOPs.

Total per training step: $\approx 6mn$ FLOPs per layer.
:::

For a Transformer with $L$ layers, hidden dimension $d$, sequence length $T$, and vocabulary size $V$, the dominant cost is the self-attention and feedforward layers. The **Chinchilla scaling law** (Hoffmann et al., 2022) gives a useful approximation:

$$C \approx 6ND$$

where $C$ is the total training compute in FLOPs, $N$ is the number of parameters, and $D$ is the number of training tokens. For a 70B parameter model trained on 1.4T tokens, this gives $C \approx 6 \times 7 \times 10^{10} \times 1.4 \times 10^{12} \approx 5.9 \times 10^{23}$ FLOPs — roughly $10^{24}$, or a "yottaFLOP."

:::{prf:remark}
:label: rmk-training-cost

The Chinchilla scaling law also prescribes the **optimal balance** between model size and data: for a fixed compute budget $C$, the optimal model size $N^*$ and data size $D^*$ both scale as $\sqrt{C}$, giving $N^* \propto D^*$. Training a model that is too large on too little data (or vice versa) wastes compute. This has practical consequences: it means that simply scaling up model size without proportionally scaling data is suboptimal — a lesson that reshaped the field's training practices.
:::

---

## GPUs and the Rise of Parallel Computing

The hardware that made deep learning practical is the **GPU** (Graphics Processing Unit), originally designed for rendering graphics but repurposed for the massively parallel matrix arithmetic that neural networks require.

:::{prf:definition} GPU Architecture (Simplified)
:label: def-gpu-architecture

A modern GPU consists of:
- **Streaming Multiprocessors (SMs)**: Each SM contains many **CUDA cores** (or equivalent) that execute arithmetic operations in parallel.
- **Tensor Cores**: Specialised units that perform small matrix multiplications (e.g., $4 \times 4$ or $8 \times 8$) in a single cycle, accelerating the dominant operation in deep learning.
- **High-Bandwidth Memory (HBM)**: DRAM attached to the GPU with high bandwidth (e.g., 3.35 TB/s on the H100) but limited capacity (e.g., 80 GB).
- **On-chip SRAM**: Small, fast memory (shared memory and L1/L2 caches) with much higher bandwidth but very limited capacity (tens of MB).
:::

The key numbers for an NVIDIA H100 (as of 2024):
- **Peak FP16 Tensor Core throughput**: ~1,979 TFLOPS (with sparsity).
- **HBM bandwidth**: 3.35 TB/s.
- **HBM capacity**: 80 GB.

The ratio of compute to memory bandwidth — the **arithmetic intensity** — determines whether an operation is **compute-bound** or **memory-bound**.

:::{prf:definition} Arithmetic Intensity and the Roofline Model
:label: def-roofline

The **arithmetic intensity** of an operation is the ratio of FLOPs to bytes transferred:

$$I = \frac{\text{FLOPs}}{\text{Bytes accessed from memory}}.$$

The **roofline model** predicts attainable performance:

$$\text{Performance} = \min(\text{Peak FLOPS},\; I \times \text{Memory bandwidth}).$$

An operation is **compute-bound** if $I > I^* = \text{Peak FLOPS} / \text{Bandwidth}$ (the "ridge point"), and **memory-bound** otherwise.
:::

For the H100, $I^* \approx 1979 \times 10^{12} / (3.35 \times 10^{12}) \approx 591$ FLOPs/byte. Large matrix multiplications (e.g., in linear layers with large batch sizes) exceed this threshold and are compute-bound. But many operations in Transformers — layer normalisation, softmax, activation functions, element-wise operations — have low arithmetic intensity and are **memory-bound**: the GPU spends most of its time waiting for data, not computing.

:::{prf:remark}
:label: rmk-flash-attention

**FlashAttention** (Dao et al., 2022) is a striking example of hardware-aware algorithm design. Standard attention computes the $T \times T$ attention matrix, which for long sequences is both memory-intensive ($O(T^2)$ space) and memory-bound (many reads/writes to HBM). FlashAttention restructures the computation using **tiling**: it processes the attention matrix in blocks that fit in on-chip SRAM, fusing the softmax, masking, and matrix multiply into a single kernel that minimises HBM accesses. The result: 2-4x faster training with no approximation — the output is numerically identical to standard attention. This is a case where understanding the hardware hierarchy (HBM → SRAM → registers) directly improves algorithmic efficiency.
:::

---

## Parallelism Strategies

A single GPU cannot hold or train a large model. Training requires distributing computation across many GPUs (hundreds to thousands). There are several complementary parallelism strategies.

:::{prf:definition} Data Parallelism
:label: def-data-parallelism

In **data parallelism**, the model is replicated across $K$ GPUs. Each GPU processes a different mini-batch $\mathcal{B}_k$ and computes local gradients $g_k = \nabla_\theta \mathcal{L}(\theta; \mathcal{B}_k)$. The gradients are then **all-reduced** (averaged) across GPUs:

$$g = \frac{1}{K} \sum_{k=1}^K g_k$$

and each GPU updates its local copy of $\theta$. This is mathematically equivalent to training with a batch size of $K \cdot |\mathcal{B}|$.
:::

Data parallelism is simple and scales well, but requires each GPU to hold the entire model — which fails for models larger than GPU memory.

:::{prf:definition} Model Parallelism
:label: def-model-parallelism

**Model parallelism** partitions the model across GPUs. Two main variants:

1. **Tensor parallelism**: A single layer's weight matrix is split across GPUs. For a linear layer $y = Wx$, partition $W$ column-wise as $W = [W_1 \mid W_2]$ across 2 GPUs, compute $y_k = W_k x$ locally, then communicate to combine. This requires fast inter-GPU communication (NVLink).

2. **Pipeline parallelism**: Different layers are assigned to different GPUs. GPU 1 computes layers 1–$L/K$, GPU 2 computes layers $L/K + 1$–$2L/K$, etc. **Micro-batching** (splitting the batch into smaller chunks that flow through the pipeline) reduces the "bubble" of idle time.
:::

:::{prf:definition} ZeRO (Zero Redundancy Optimiser)
:label: def-zero

**ZeRO** (Rajbhandari et al., 2020) reduces memory redundancy in data parallelism. Standard data parallelism replicates three things on each GPU: model parameters, gradients, and optimiser states (e.g., Adam's first and second moments). ZeRO partitions these across GPUs:

- **Stage 1**: Partition optimiser states → ~4x memory reduction.
- **Stage 2**: Partition gradients → ~8x memory reduction.
- **Stage 3**: Partition parameters → ~$K$x memory reduction (linear in GPU count).

Each GPU holds only a shard and communicates with others when needed. This allows data-parallel training of models that no single GPU can hold.
:::

In practice, large-scale training combines all three: ZeRO or FSDP (Fully Sharded Data Parallel) for memory efficiency, tensor parallelism within a node (where NVLink provides high bandwidth), and pipeline parallelism across nodes.

---

## Quantisation

Reducing the numerical precision of weights and activations — **quantisation** — is one of the most effective techniques for efficient inference.

:::{prf:definition} Quantisation
:label: def-quantisation

**Quantisation** maps floating-point values to a lower-precision representation. For **uniform symmetric quantisation** to $b$ bits:

$$Q(x) = \text{round}\!\left(\frac{x}{s}\right) \cdot s, \quad s = \frac{\max|x|}{2^{b-1} - 1}$$

where $s$ is the **scale factor**. Common precisions:
- **FP32** (32 bits): Full precision. Standard for scientific computing.
- **FP16 / BF16** (16 bits): Half precision. Standard for training. BF16 has more exponent bits (8 vs 5), better dynamic range, less precision — preferred for deep learning.
- **INT8** (8 bits): Post-training quantisation for inference. ~2x memory reduction, ~2x throughput increase.
- **INT4** (4 bits): Aggressive quantisation. ~4x memory reduction with moderate quality loss.
:::

:::{prf:remark}
:label: rmk-quantisation-theory

The quality of quantised models depends on the weight distribution. Weights with large outliers are harder to quantise because the scale factor $s$ must accommodate the outliers, leaving most of the quantisation grid unused for the bulk of (smaller) values. Methods like **GPTQ** (Frantar et al., 2022) and **AWQ** (Lin et al., 2023) address this by quantising weights in a data-aware fashion: they minimise the quantisation error on a calibration dataset rather than quantising each weight independently. The connection to [LoRA](8_post_training.md) is natural: if fine-tuning updates have low rank, the quantisation error on the fine-tuned model can be compensated by a small LoRA adapter — the **QLoRA** approach (Dettmers et al., 2023), which enables fine-tuning 65B models on a single 48GB GPU.
:::

---

## Memory and the Inference Bottleneck

During **training**, the bottleneck is compute: matrix multiplications dominate. During **inference** (especially autoregressive generation), the bottleneck shifts to **memory bandwidth**.

:::{prf:definition} Autoregressive Inference Cost
:label: def-inference-cost

Generating one token from a Transformer with $N$ parameters requires reading approximately $2N$ bytes (in FP16) from memory — the entire model. The generation rate is therefore bounded by:

$$\text{Tokens/second} \leq \frac{\text{Memory bandwidth (bytes/s)}}{2N}$$

For a 70B parameter model in FP16 on an H100: $3.35 \times 10^{12} / (2 \times 7 \times 10^{10}) \approx 24$ tokens/second per GPU. This is **memory-bound**: the Tensor Cores are almost entirely idle.
:::

This is why quantisation has such a large impact on inference speed: reducing from FP16 to INT4 cuts the memory footprint by 4x, directly increasing the generation rate by the same factor (assuming the operation remains memory-bound).

**KV-cache** management is another critical inference concern. During autoregressive generation, each token's key and value vectors from every layer must be stored for use by subsequent tokens. For a model with $L$ layers, $H$ attention heads, and head dimension $d_h$, generating a sequence of length $T$ requires:

$$\text{KV-cache size} = 2 \times L \times H \times d_h \times T \times \text{bytes per element}.$$

For a 70B model (80 layers, 64 heads, $d_h = 128$) generating 8192 tokens in FP16: $2 \times 80 \times 64 \times 128 \times 8192 \times 2 \approx 21$ GB — a significant fraction of GPU memory, and it grows linearly with sequence length.

---

## Specialised Hardware

The limitations of general-purpose GPUs have spurred the development of specialised AI accelerators.

**Google TPUs** (Tensor Processing Units) are ASICs designed specifically for matrix multiplication. TPU v4 provides 275 TFLOPS of BF16 compute with 1.2 TB/s HBM bandwidth, interconnected via a custom **ICI** (Inter-Chip Interconnect) that enables efficient all-reduce across thousands of chips. TPU pods (thousands of chips connected in a torus topology) can train the largest models.

**Emerging architectures** explore different points in the design space:
- **Wafer-scale computing** (Cerebras CS-3): An entire wafer as a single chip with 900,000 cores, 44 GB of on-chip SRAM, and no off-chip memory bottleneck. Eliminates the HBM bandwidth wall entirely for models that fit on-chip.
- **Near-memory computing**: Move compute closer to memory (or onto the memory die) to reduce data movement — the dominant source of both latency and energy consumption.
- **Photonic computing**: Use light instead of electrons for matrix multiplication, potentially achieving orders-of-magnitude improvements in energy efficiency.

---

## Scaling Laws and Compute Governance

The empirical **scaling laws** (Kaplan et al., 2020; Hoffmann et al., 2022) quantify how model performance improves with compute, data, and parameters.

:::{prf:definition} Neural Scaling Laws
:label: def-scaling-laws

The cross-entropy loss $L$ of a language model scales as a power law in the key resources:

$$L(N) \approx \left(\frac{N_c}{N}\right)^{\alpha_N}, \quad L(D) \approx \left(\frac{D_c}{D}\right)^{\alpha_D}, \quad L(C) \approx \left(\frac{C_c}{C}\right)^{\alpha_C}$$

where $N$ = parameters, $D$ = data tokens, $C$ = compute FLOPs, and $N_c, D_c, C_c, \alpha_N, \alpha_D, \alpha_C$ are empirical constants. Typical values: $\alpha_N \approx 0.076$, $\alpha_D \approx 0.095$, $\alpha_C \approx 0.050$ (Kaplan et al., 2020).
:::

These power laws have profound implications:

1. **Diminishing returns**: Each 10x increase in compute yields a roughly constant absolute improvement in loss. Achieving human-level performance on a benchmark that is 0.1 nats away requires $10^{0.1/0.05} = 100$x more compute.

2. **Predictability**: Performance at scale can be predicted from smaller-scale experiments, enabling rational allocation of multi-million-dollar training budgets.

3. **Compute as the binding constraint**: Given sufficient data, performance is determined by compute. This has led to "compute governance" proposals — regulating access to large-scale compute as a lever for AI safety policy.

:::{prf:remark}
:label: rmk-hardware-lottery

Hooker (2021) argued that the success of particular architectures (CNNs, Transformers) is partly a **hardware lottery**: these architectures happen to map well onto the hardware that was available (GPUs optimised for dense matrix multiplication). Architectures that require different computational primitives — sparse computation, symbolic manipulation, neuromorphic processing — may be underexplored not because they are inferior but because the hardware to run them efficiently does not yet exist. The interplay between hardware and algorithms is a co-evolutionary process: hardware shapes which algorithms are practical, and successful algorithms shape the next generation of hardware.
:::
