# Post-Training

Pre-training a large language model on internet text produces a system that is remarkably capable but not particularly useful: it generates plausible continuations of text, but doesn't follow instructions, answer questions helpfully, or avoid harmful outputs. **Post-training** is the process of transforming a pre-trained model into one that is actually useful, safe, and aligned with human intent.

This chapter covers the key post-training techniques: **supervised fine-tuning (SFT)**, **reinforcement learning from human feedback (RLHF)**, **direct preference optimisation (DPO)**, and **parameter-efficient fine-tuning (PEFT)**. These methods are responsible for the difference between a raw language model and an AI assistant.

---

## The Post-Training Pipeline

The standard post-training pipeline has three stages:

1. **Pre-training**: Train a large model on a massive corpus of text via next-token prediction. This produces a "base model" — a strong language model with broad knowledge but no specific behaviour.
2. **Supervised Fine-Tuning (SFT)**: Fine-tune on a curated dataset of (instruction, response) pairs. This teaches the model to follow instructions and produce helpful outputs.
3. **Alignment**: Further refine the model to match human preferences, typically via RLHF or DPO. This improves quality, reduces harmfulness, and enhances instruction-following.

---

## Supervised Fine-Tuning

:::{prf:definition} Supervised Fine-Tuning
:label: def-sft

Given a pre-trained model $p_\theta$ and a dataset of demonstrations $\mathcal{D}_{\text{SFT}} = \{(\text{instruction}_i, \text{response}_i)\}_{i=1}^n$, **SFT** minimises the conditional cross-entropy loss:

$$\mathcal{L}_{\text{SFT}}(\theta) = -\sum_{i=1}^n \sum_{t=1}^{T_i} \log p_\theta(y_t^{(i)} \mid \text{instruction}_i, y_{<t}^{(i)})$$

where $y^{(i)} = (y_1^{(i)}, \ldots, y_{T_i}^{(i)})$ is the $i$-th response tokenised.
:::

SFT is conceptually simple — it is just language modelling on a curated dataset — but the data quality matters enormously. A small dataset of high-quality demonstrations (thousands to tens of thousands of examples) can dramatically shift the model's behaviour from "predict likely text" to "follow instructions and provide helpful answers."

The SFT stage teaches the model the *format* of helpful behaviour: how to structure responses, when to refuse, how to express uncertainty. The subsequent alignment stage teaches the model to distinguish *good* from *bad* responses within this format.

---

## Reinforcement Learning from Human Feedback

**RLHF** (Christiano et al., 2017; Ouyang et al., 2022) uses human preference data to train a reward model, then uses RL to optimise the language model against this reward.

### Step 1: Reward Model Training

:::{prf:definition} Reward Model
:label: def-reward-model

A **reward model** $r_\psi: \mathcal{X} \times \mathcal{Y} \to \mathbb{R}$ maps (prompt, response) pairs to scalar scores. It is trained on human preference data $\mathcal{D}_{\text{pref}} = \{(x_i, y_i^w, y_i^l)\}_{i=1}^n$ where $y^w$ is preferred over $y^l$ by a human annotator.

The reward model is trained via the **Bradley-Terry** preference model:

$$\mathcal{L}_{\text{RM}}(\psi) = -\sum_{i=1}^n \log \sigma\!\left(r_\psi(x_i, y_i^w) - r_\psi(x_i, y_i^l)\right)$$

where $\sigma$ is the sigmoid function. This loss pushes the reward model to assign higher scores to preferred responses.
:::

### Step 2: Policy Optimisation

:::{prf:definition} RLHF Objective
:label: def-rlhf-objective

Given the reward model $r_\psi$ and a reference policy $\pi_{\text{ref}}$ (typically the SFT model), RLHF optimises:

$$\max_\theta \; \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi_\theta(\cdot \mid x)}\!\left[r_\psi(x, y) - \beta \, D_{\text{KL}}(\pi_\theta(\cdot \mid x) \| \pi_{\text{ref}}(\cdot \mid x))\right]$$

where $\beta > 0$ controls the **KL penalty** — a regularisation that prevents the model from drifting too far from the reference.
:::

The KL penalty is crucial: without it, the policy would exploit the reward model, finding adversarial outputs that score high on $r_\psi$ but are actually nonsensical (reward hacking). The penalty keeps the policy close to the well-behaved SFT model.

In practice, this optimisation is performed using **PPO** (from the [Reinforcement Learning](7_reinforcement_learning.md) chapter). The language model is the policy, the prompt is the state, the generated response is the action, and the reward model score (minus KL penalty) is the reward.

---

## Direct Preference Optimisation

**DPO** (Rafailov et al., 2023) is an elegant alternative to RLHF that eliminates the need for a separate reward model and the RL training loop entirely.

:::{prf:theorem} DPO Equivalence
:label: thm-dpo

The optimal policy for the RLHF objective (with KL regularisation) has the closed-form:

$$\pi^*(y \mid x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y \mid x) \exp\!\left(\frac{1}{\beta} r(x, y)\right)$$

where $Z(x)$ is a normalising partition function. This implies:

$$r(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x).$$

Substituting this into the Bradley-Terry preference model and cancelling $Z(x)$ (which is constant for a given prompt), the preference loss becomes a function of the policy alone.
:::

:::{prf:definition} DPO Loss
:label: def-dpo

The **DPO loss** directly optimises the policy on preference data:

$$\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x, y^w, y^l)}\!\left[\log \sigma\!\left(\beta \log \frac{\pi_\theta(y^w \mid x)}{\pi_{\text{ref}}(y^w \mid x)} - \beta \log \frac{\pi_\theta(y^l \mid x)}{\pi_{\text{ref}}(y^l \mid x)}\right)\right].$$
:::

DPO is remarkable: it achieves the same theoretical optimum as RLHF but with a simple supervised loss — no reward model, no RL, no PPO. The log-ratio $\log(\pi_\theta / \pi_{\text{ref}})$ acts as an *implicit* reward. Training is stable and efficient.

The intuition: the loss increases the log-probability of preferred responses and decreases the log-probability of dispreferred responses, with the reference model providing a baseline that prevents collapse.

---

## Parameter-Efficient Fine-Tuning

Full fine-tuning updates all parameters of the model, which for large models (billions of parameters) is expensive in compute and memory. **Parameter-efficient fine-tuning (PEFT)** methods update only a small number of additional parameters while keeping the base model frozen.

:::{prf:definition} Low-Rank Adaptation (LoRA)
:label: def-lora

**LoRA** (Hu et al., 2021) modifies each weight matrix $W \in \mathbb{R}^{d \times k}$ by adding a low-rank perturbation:

$$W' = W + \Delta W = W + BA$$

where $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times k}$ with $r \ll \min(d, k)$. Only $B$ and $A$ are trained; $W$ is frozen.

The number of trainable parameters per layer is $r(d + k)$ instead of $dk$ — a reduction factor of approximately $dk / (r(d+k)) \approx d / (2r)$ for square matrices.
:::

LoRA is motivated by the hypothesis that the weight updates during fine-tuning have low intrinsic rank — the model doesn't need to change in all directions, only along a low-dimensional subspace relevant to the task.

:::{prf:remark}
:label: rmk-metatt

Our own work on [MetaTT](https://arxiv.org/pdf/2506.09105) (2025) extends the PEFT idea using **tensor-train decomposition** instead of low-rank factorisation. Tensor trains can achieve even higher compression ratios ($\sim$10x compared to LoRA) with comparable downstream performance, by exploiting the multi-dimensional structure of weight tensors in Transformers. The key insight is that the weight tensor naturally decomposes along multiple modes (input, output, head, layer), and a tensor-train captures correlations across all of these simultaneously.
:::

Other PEFT methods include:
- **Adapters**: Small bottleneck layers inserted between existing layers.
- **Prefix tuning**: Prepend learnable "virtual tokens" to the input.
- **Prompt tuning**: Learn continuous embeddings for prompt tokens.

---

## Knowledge Distillation

**Distillation** (Hinton et al., 2015) transfers knowledge from a large **teacher** model to a smaller **student** model.

:::{prf:definition} Knowledge Distillation
:label: def-distillation

Given a teacher model $p_T$ and a student model $p_S^\theta$, distillation minimises a mixture of the hard-label loss and the **soft-label loss** (matching the teacher's output distribution):

$$\mathcal{L}_{\text{distill}}(\theta) = (1 - \alpha) \mathcal{L}_{\text{CE}}(y, p_S^\theta(x)) + \alpha \, T^2 \, D_{\text{KL}}(p_T^{(\tau)}(x) \| p_S^{\theta, (\tau)}(x))$$

where $p^{(\tau)}$ denotes the softmax output with temperature $\tau$: $p_i^{(\tau)} = \exp(z_i/\tau) / \sum_j \exp(z_j/\tau)$, and $\alpha$ balances the two losses.
:::

High temperature ($\tau > 1$) softens the teacher's output distribution, revealing the relative probabilities of non-top classes — the "dark knowledge" that is lost when we only look at the argmax. A student trained to match these soft distributions learns more than one trained on hard labels alone.

Distillation is widely used to create smaller, faster models for deployment. It is also used in the post-training pipeline itself: a strong model can generate training data for a weaker model, effectively bootstrapping alignment.
