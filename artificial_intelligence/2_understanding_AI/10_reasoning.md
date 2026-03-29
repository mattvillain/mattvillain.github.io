# Reasoning

A language model that can fluently discuss quantum mechanics is not necessarily *reasoning* about quantum mechanics. It may be pattern-matching against its training data, producing text that *looks* like reasoning without performing the underlying logical steps. The distinction matters: a model that truly reasons can solve novel problems, while one that merely retrieves can only solve problems similar to those it has seen.

This chapter surveys the emerging science of reasoning in AI systems: what it means, how it is elicited, and where the current frontiers lie.

---

## What Is Reasoning?

Reasoning is the process of drawing conclusions from premises via a sequence of logical steps. For AI systems, we can distinguish several forms:

- **Deductive reasoning**: Given premises, derive necessary conclusions. "All men are mortal; Socrates is a man; therefore Socrates is mortal."
- **Inductive reasoning**: Given observations, infer general patterns. "Every swan I have seen is white; therefore swans are white."
- **Abductive reasoning**: Given an observation, infer the best explanation. "The ground is wet; it probably rained."
- **Mathematical reasoning**: Construct proofs, solve equations, verify formal statements.
- **Commonsense reasoning**: Apply everyday knowledge that humans take for granted. "If I drop a glass, it will break."

Language models trained on next-token prediction acquire some capacity for all of these — but the depth and reliability of that capacity is the central question.

---

## Chain-of-Thought Prompting

The simplest intervention that dramatically improves reasoning performance is asking the model to "think step by step."

:::{prf:definition} Chain-of-Thought Prompting
:label: def-cot

**Chain-of-thought (CoT) prompting** (Wei et al., 2022) augments a prompt with intermediate reasoning steps before the final answer. Given a question $x$ and answer $y$, CoT generates a reasoning trace $z = (z_1, z_2, \ldots, z_T)$ such that:

$$p(y \mid x) = \sum_z p(y \mid z, x) \, p(z \mid x).$$

In practice, the model generates $z$ autoregressively, then conditions on $z$ to produce $y$.
:::

The effect is striking: on the GSM8K benchmark (grade-school math), CoT prompting improved accuracy from 17.9% to 58.1% for PaLM 540B (Wei et al., 2022). The reasoning trace decomposes a complex problem into manageable subproblems, each of which is within the model's capability.

**Why does CoT work?** Several hypotheses:

1. **Computation depth**: A Transformer with $L$ layers performs $O(L)$ serial computation steps per token. By generating intermediate tokens, CoT effectively increases the computational depth — each token in the chain adds another $L$ steps of processing. The model "thinks longer" on harder problems.

2. **Working memory**: The reasoning trace acts as an external scratchpad. Intermediate results are written to the context window, freeing the model from having to maintain them in its fixed-dimensional hidden state.

3. **Training distribution**: The pre-training data contains many examples of step-by-step reasoning (textbooks, solutions manuals, forum posts). CoT prompting activates this learned pattern.

:::{prf:remark}
:label: rmk-cot-faithfulness

A critical open question: are chain-of-thought traces **faithful** — i.e., do they reflect the model's actual reasoning process? Or are they post-hoc rationalisations that sound plausible but do not causally influence the output? Evidence is mixed. Lanham et al. (2023) showed that in some cases, models produce correct answers even when the CoT trace contains errors, and that truncating the CoT sometimes does not change the answer. This connects directly to the interpretability challenges discussed in the [Interpretability](9_interpretability.md) chapter: understanding *what* a model computes internally remains difficult even when it verbalises a reasoning process.
:::

---

## Test-Time Compute and Search

A fundamental insight: reasoning can be improved by spending more computation at **inference time**, not just at training time. This is the paradigm of **test-time compute scaling**.

:::{prf:definition} Test-Time Compute Scaling
:label: def-test-time-compute

**Test-time compute scaling** refers to methods that improve model performance by allocating additional computation during inference. The key mechanisms include:

1. **Extended generation**: Generate longer reasoning traces (more tokens of "thinking").
2. **Sampling and selection**: Generate multiple candidate solutions and select the best one.
3. **Search**: Explore a tree of reasoning paths and use a verifier to guide the search.
:::

### Best-of-N Sampling

The simplest test-time scaling method: generate $N$ independent solutions and select the one that a verifier (or the model itself) judges to be best.

:::{prf:definition} Best-of-N with Verifier
:label: def-best-of-n

Given a generator $\pi_\theta$ and a verifier $V_\psi$, the **best-of-$N$** strategy produces:

$$y^* = \arg\max_{y \in \{y_1, \ldots, y_N\}} V_\psi(x, y), \quad y_i \sim \pi_\theta(\cdot \mid x).$$

If each candidate independently has probability $p$ of being correct, the probability that at least one of $N$ candidates is correct is $1 - (1-p)^N$, which approaches 1 as $N$ grows.
:::

For mathematical reasoning, the verifier is often a **reward model** trained on human preferences (connecting to [Post-Training](8_post_training.md)) or, more powerfully, an **outcome-based verifier** that checks whether the final answer is correct.

### Tree Search

More sophisticated: rather than generating complete solutions independently, explore a **tree** of partial reasoning steps.

:::{prf:definition} Monte Carlo Tree Search for Reasoning
:label: def-mcts-reasoning

Given a reasoning problem $x$, construct a tree where:
- **Nodes** are partial reasoning traces $(z_1, \ldots, z_k)$.
- **Edges** are individual reasoning steps $z_{k+1}$.
- **Rollouts** complete partial traces to final answers.
- A **value function** $V(z_1, \ldots, z_k)$ estimates the probability that the partial trace leads to a correct answer.

**MCTS** (from the [Reinforcement Learning](7_reinforcement_learning.md) chapter) balances exploration and exploitation using UCB:

$$\text{UCB}(z) = \bar{V}(z) + c \sqrt{\frac{\ln N(\text{parent}(z))}{N(z)}}$$

where $\bar{V}(z)$ is the mean value of rollouts through node $z$, $N(z)$ is the visit count, and $c$ controls exploration.
:::

This approach — used in systems like AlphaProof and reasoning-focused models — transforms language model inference from a single forward pass into a deliberate search process. The model generates candidate reasoning steps, evaluates them, backtracks from dead ends, and explores alternatives. This is much closer to how humans solve hard problems: not in a single stream of consciousness, but by trying approaches, checking them, and revising.

:::{prf:remark}
:label: rmk-test-time-scaling-law

Snell et al. (2024) demonstrated that test-time compute scaling follows its own scaling laws: performance improves log-linearly with the amount of test-time compute (number of samples, search depth), and for sufficiently hard problems, spending compute at test time can be more efficient than spending it on training. This suggests an optimal allocation between training-time and test-time compute that depends on the difficulty distribution of the target tasks.
:::

---

## Process Reward Models

A key component of reasoning systems is the **process reward model** (PRM): a model that evaluates individual reasoning steps, not just final answers.

:::{prf:definition} Process Reward Model
:label: def-prm

An **outcome reward model** (ORM) scores complete solutions: $r_{\text{ORM}}(x, y) \in \mathbb{R}$.

A **process reward model** (PRM) scores individual reasoning steps: $r_{\text{PRM}}(x, z_1, \ldots, z_k) \in \mathbb{R}$ for each step $k$.

The PRM provides denser feedback: rather than waiting until the end to learn whether the solution was correct, the model receives a signal at each step indicating whether the reasoning is on track.
:::

:::{prf:remark}
:label: rmk-prm-vs-orm

Lightman et al. (2023) showed that PRMs significantly outperform ORMs for guiding search on mathematical reasoning tasks. The intuition: an ORM can only distinguish correct from incorrect complete solutions, while a PRM can identify the *first* incorrect step and redirect search before the error propagates. This is analogous to the difference between sparse and dense reward signals in reinforcement learning — dense rewards (PRMs) make learning much more efficient.
:::

Training PRMs requires step-level annotations, which are expensive to collect from humans. An alternative is to train PRMs automatically using **Monte Carlo estimation**: for each reasoning step, sample many completions and estimate the step's quality by the fraction that reach a correct final answer. This connects directly to the Monte Carlo methods in the [Numerical Methods](../../mathematics_for_quantitative_finance/5_numerical_methods.md) chapter.

---

## Formal Reasoning and Theorem Proving

Mathematical theorem proving is the purest form of reasoning: every step must be logically valid, and the final result is either correct or incorrect — there is no ambiguity.

:::{prf:definition} Automated Theorem Proving
:label: def-atp

An **automated theorem prover** takes a formal statement $\phi$ in a logical language (e.g., Lean 4, Coq, Isabelle) and attempts to construct a proof $\pi$ such that $\vdash \pi : \phi$ — i.e., $\pi$ is a valid proof of $\phi$ according to the type-checking rules of the formal system.

A **neural theorem prover** uses a language model to generate proof steps (tactics), with the formal system acting as a verifier:

$$a_t \sim \pi_\theta(\cdot \mid \phi, s_t)$$

where $s_t$ is the current proof state (remaining goals) and $a_t$ is the next tactic to apply.
:::

This is a natural application of RL and search: the proof state is the environment state, tactics are actions, and successful proof completion is the reward. Systems like AlphaProof (DeepMind, 2024) combined language models with MCTS to solve olympiad-level mathematics problems — a landmark result.

:::{prf:remark}
:label: rmk-formal-verification

Formal theorem proving has a unique advantage for AI safety: proofs can be **mechanically verified**. Unlike natural-language reasoning, where we must trust (or interpret) the model's chain of thought, a formal proof either type-checks or it doesn't. This makes formal reasoning a promising avenue for building **provably correct** AI systems — though the gap between formal and informal reasoning remains large.
:::

---

## Reasoning as Reinforcement Learning

There is a deep connection between reasoning and reinforcement learning. Generating a chain of thought is a sequential decision process: at each step, the model chooses what to write next, and the quality of the final answer depends on the entire sequence of choices.

:::{prf:remark}
:label: rmk-reasoning-rl

The reasoning problem can be formalised as an MDP (see [Reinforcement Learning](7_reinforcement_learning.md)):

- **State** $s_t$: the prompt $x$ concatenated with the reasoning trace so far $(z_1, \ldots, z_t)$.
- **Action** $a_t$: the next reasoning step $z_{t+1}$ (or the next token).
- **Transition**: deterministic — the new state appends $a_t$ to the trace.
- **Reward**: sparse (correctness of final answer) or dense (PRM score at each step).

Training the model to reason well is then a policy optimisation problem: maximise the expected reward of the reasoning traces. This is why RLHF and RL-based training (PPO, GRPO) are central to reasoning model development — they directly optimise the sequential decision-making process that reasoning requires.
:::

**GRPO** (Group Relative Policy Optimisation; Shao et al., 2024) is a variant particularly suited to reasoning: it samples a group of responses, computes their relative rewards, and uses the group statistics as a baseline — eliminating the need for a separate critic network that PPO requires. This simplifies training while maintaining the benefits of policy optimisation.

---

## The Frontier: What Can Models Reason About?

Current reasoning models achieve remarkable performance on mathematical competitions, coding challenges, and scientific problems. But fundamental questions remain:

1. **Generalisation**: Can models reason about truly novel problems, or do they require the reasoning patterns to be well-represented in the training data? Evidence suggests that performance degrades sharply on problems that differ structurally from training examples.

2. **Compositionality**: Can models combine learned reasoning primitives in new ways? Compositional generalisation — applying known rules in novel combinations — remains challenging.

3. **Faithfulness**: When a model produces a correct answer with a plausible reasoning trace, is the trace the actual cause of the answer? If not, the "reasoning" is a post-hoc narrative, and we cannot rely on it for alignment or safety purposes.

4. **Theoretical limits**: What are the fundamental computational limits of autoregressive reasoning? A Transformer generating $T$ tokens of chain-of-thought can simulate a Turing machine for $T$ steps (given sufficient width). But practical models operate far from this theoretical limit.

These questions are at the frontier of AI research, and their answers will shape the next generation of AI systems.
