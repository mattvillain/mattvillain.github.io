# Reinforcement Learning

In supervised learning, a model is given labelled examples and asked to predict labels on new inputs. In reinforcement learning, an **agent** interacts with an **environment**, takes **actions**, receives **rewards**, and must learn a **policy** — a strategy for choosing actions — that maximises cumulative reward over time.

This is a fundamentally different learning paradigm. There are no labelled examples. The agent learns by trial and error: it takes an action, observes the consequence, and updates its strategy. The reward signal is often sparse and delayed — the agent may not know for many steps whether an action was good or bad. The environment is typically stochastic and partially observable. Despite these challenges, reinforcement learning has produced some of the most spectacular results in AI: mastering Go (AlphaGo), Atari games (DQN), robotic control, and — crucially for modern language models — aligning AI systems with human preferences (RLHF).

---

## Markov Decision Processes

The mathematical foundation of RL is the **Markov Decision Process** (MDP).

:::{prf:definition} Markov Decision Process
:label: def-mdp

A **Markov Decision Process** is a tuple $(\mathcal{S}, \mathcal{A}, P, R, \gamma)$ where:

- $\mathcal{S}$: the **state space**.
- $\mathcal{A}$: the **action space**.
- $P: \mathcal{S} \times \mathcal{A} \to \Delta(\mathcal{S})$: the **transition kernel** — $P(s' \mid s, a)$ is the probability of moving to state $s'$ given state $s$ and action $a$.
- $R: \mathcal{S} \times \mathcal{A} \to \mathbb{R}$: the **reward function** — $R(s, a)$ is the immediate reward for taking action $a$ in state $s$.
- $\gamma \in [0, 1)$: the **discount factor**, trading off immediate and future rewards.
:::

The **Markov property** — the transition probabilities depend only on the current state and action, not on history — is the key structural assumption. It makes the problem tractable.

:::{prf:definition} Policy and Value Functions
:label: def-policy-value

A **policy** $\pi: \mathcal{S} \to \Delta(\mathcal{A})$ maps states to distributions over actions. The **state-value function** under policy $\pi$ is:

$$V^\pi(s) = \mathbb{E}_\pi\!\left[\sum_{t=0}^\infty \gamma^t R(s_t, a_t) \;\middle|\; s_0 = s\right].$$

The **action-value (Q) function** is:

$$Q^\pi(s, a) = \mathbb{E}_\pi\!\left[\sum_{t=0}^\infty \gamma^t R(s_t, a_t) \;\middle|\; s_0 = s, a_0 = a\right].$$

The **optimal policy** $\pi^*$ satisfies $V^{\pi^*}(s) \geq V^\pi(s)$ for all $s$ and all $\pi$.
:::

---

## The Bellman Equations

The value functions satisfy recursive relationships — the **Bellman equations** — which are the RL analogues of dynamic programming.

:::{prf:theorem} Bellman Equations
:label: thm-bellman

The value function $V^\pi$ satisfies:

$$V^\pi(s) = \sum_{a} \pi(a \mid s) \left[R(s, a) + \gamma \sum_{s'} P(s' \mid s, a) V^\pi(s')\right].$$

The optimal value function $V^* = V^{\pi^*}$ satisfies the **Bellman optimality equation**:

$$V^*(s) = \max_a \left[R(s, a) + \gamma \sum_{s'} P(s' \mid s, a) V^*(s')\right].$$

Similarly, the optimal Q-function satisfies:

$$Q^*(s, a) = R(s, a) + \gamma \sum_{s'} P(s' \mid s, a) \max_{a'} Q^*(s', a').$$
:::

The optimal policy is recovered greedily: $\pi^*(s) = \arg\max_a Q^*(s, a)$. If we can compute $Q^*$, we have solved the problem.

---

## Value-Based Methods

**Value-based methods** learn the value function (or Q-function) and derive the policy from it.

### Q-Learning

:::{prf:definition} Q-Learning (Watkins, 1989)
:label: def-q-learning

**Q-learning** is an off-policy algorithm that learns $Q^*$ directly by iterating:

$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[r_t + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t)\right]$$

where $\alpha$ is the learning rate and the bracketed term is the **temporal difference (TD) error**.
:::

:::{prf:theorem} Q-Learning Convergence
:label: thm-q-convergence

Under standard conditions (all state-action pairs visited infinitely often, learning rate schedule $\sum_t \alpha_t = \infty$ and $\sum_t \alpha_t^2 < \infty$), Q-learning converges to $Q^*$ almost surely.
:::

### Deep Q-Networks (DQN)

For large or continuous state spaces, tabular Q-learning is infeasible. **Deep Q-Networks** (Mnih et al., 2015) approximate $Q^*$ with a neural network $Q_\theta(s, a)$ trained by minimising:

$$\mathcal{L}(\theta) = \mathbb{E}\!\left[\left(r + \gamma \max_{a'} Q_{\theta^-}(s', a') - Q_\theta(s, a)\right)^2\right]$$

where $\theta^-$ is a **target network** (a periodically updated copy of $\theta$) that stabilises training. DQN also uses an **experience replay buffer** — a dataset of past transitions $(s, a, r, s')$ — to break temporal correlations in the training data.

DQN achieved human-level performance on Atari games and demonstrated that deep RL could scale to high-dimensional observation spaces (raw pixels).

---

## Policy-Based Methods

**Policy-based methods** parameterise the policy directly as $\pi_\theta(a \mid s)$ and optimise $\theta$ to maximise expected cumulative reward.

:::{prf:theorem} Policy Gradient Theorem (Sutton et al., 1999)
:label: thm-policy-gradient

The gradient of the expected return $J(\theta) = \mathbb{E}_{\pi_\theta}[\sum_t \gamma^t R(s_t, a_t)]$ with respect to $\theta$ is:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\!\left[\sum_{t=0}^\infty \gamma^t \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot Q^{\pi_\theta}(s_t, a_t)\right].$$
:::

:::{prf:proof}
:class: dropdown
:enumerated: false

The key identity: $\nabla_\theta \pi_\theta(a \mid s) = \pi_\theta(a \mid s) \nabla_\theta \log \pi_\theta(a \mid s)$ (the "log-derivative trick"). Starting from $J(\theta) = \sum_s d^{\pi_\theta}(s) \sum_a \pi_\theta(a \mid s) Q^{\pi_\theta}(s, a)$ where $d^{\pi_\theta}$ is the discounted state distribution, differentiate with respect to $\theta$. The gradient of $d^{\pi_\theta}$ is handled by the policy gradient theorem, which shows it can be absorbed into the expectation:

$$\nabla_\theta J(\theta) = \sum_s d^{\pi_\theta}(s) \sum_a \nabla_\theta \pi_\theta(a \mid s) Q^{\pi_\theta}(s, a) = \mathbb{E}_{\pi_\theta}[\nabla_\theta \log \pi_\theta(a \mid s) Q^{\pi_\theta}(s, a)]. \quad \square$$
:::

The **REINFORCE** algorithm (Williams, 1992) estimates this gradient by sampling trajectories and replacing $Q^{\pi_\theta}$ with the observed return. This is unbiased but has high variance.

---

## Actor-Critic Methods

**Actor-critic** methods combine the best of both worlds: a parameterised policy $\pi_\theta$ (the actor) and a learned value function $V_\phi$ or $Q_\phi$ (the critic).

:::{prf:definition} Advantage Actor-Critic (A2C)
:label: def-a2c

The **advantage** of action $a$ in state $s$ is $A^{\pi}(s, a) = Q^{\pi}(s, a) - V^{\pi}(s)$: how much better action $a$ is compared to the average. The actor-critic update is:

- **Critic update**: Minimise $\mathbb{E}[(r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t))^2]$ (TD error).
- **Actor update**: $\theta \leftarrow \theta + \alpha \nabla_\theta \log \pi_\theta(a_t \mid s_t) \hat{A}_t$ where $\hat{A}_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$.
:::

Using the advantage instead of the raw Q-value reduces variance (since $V^{\pi}$ acts as a baseline). **Proximal Policy Optimisation (PPO)** (Schulman et al., 2017) adds a clipped surrogate objective that prevents the policy from changing too much in a single update, improving training stability:

$$\mathcal{L}^{\text{PPO}}(\theta) = \mathbb{E}\!\left[\min\!\left(\frac{\pi_\theta(a \mid s)}{\pi_{\theta_{\text{old}}}(a \mid s)} \hat{A},\; \text{clip}\!\left(\frac{\pi_\theta}{\pi_{\theta_{\text{old}}}}, 1-\epsilon, 1+\epsilon\right) \hat{A}\right)\right].$$

PPO is the workhorse algorithm behind RLHF (Reinforcement Learning from Human Feedback), which we will discuss in the [Post-Training](8_post_training.md) chapter.

---

## Connections

Reinforcement learning connects to several other topics in this book:

- **Optimal stopping** (American options, ch. 6 of Quantitative Finance): Exercising an American option is an RL problem where the action space is $\{\text{exercise}, \text{continue}\}$ and the reward is the discounted payoff. The Bellman equation is the Snell envelope recursion.

- **Portfolio management** (ch. 9 of Quantitative Finance): Portfolio rebalancing is a continuous-action MDP where states are market features and actions are portfolio weights.

- **Post-training of language models** (next chapter): RLHF uses PPO to fine-tune a language model policy to maximise a learned reward model of human preferences.

- **Category theory**: An MDP can be formalised as a morphism in a Markov category, with the transition kernel as a stochastic morphism $P: \mathcal{S} \times \mathcal{A} \to \mathcal{S}$. The value function is a fixed point of the Bellman operator, which has a categorical characterisation as an algebra for a functor.
