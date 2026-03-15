# 2. Probability for Finance

There are millions of course on probability theory. Therefore, I will be slighlty esotic - and if you do not like what you see, feel free to return to one of the many great guides on studying probability. 

Probabilities are models of uncertainty. The world is unbearably complex; often, instead of attempting to compute extremely complex phenomena (such as the trajectory of the toss of a coin, or the exact dynamics in shuffling a deck of cards), we make simplifying, yet useful, assumptions. This is epistemic uncertainty. Moreover, certain phenomena are intrinsically probabilistic - such as those occuring at the quantum scale - but I know nothing about those. Those would be aleatoric uncertainty. Aleatoric just means random (comes from *alea* which is dice in Latin, like *Alea iacta Est*, as Julius Caesar famously said).

There's two key components in defining a notion of probability: the notion of events - and being able to discern between them precisely and formally - and the measure of probability. This is how it usually goes: 

:::{prf:definition} Probability Space
:label: def-probability-space

A probability space is a triple $(\Omega,\mathcal{F},\mathbb{P})$, composed of: 
- $\Omega : \textbf{Set}$ the space of all events. This is a set, can be finite or infinite and can be thought of atomic, individual events. It can be a numerical set too, like $\mathbb{N}, \mathbb{R}$. 
- $\mathcal{F}: 2^{\Omega}$ is a set of subsets of $\Omega$ called a $\sigma$-algebra. A $\sigma$-algebra has the following properties: 
    - **Closure under Intersection**: $A,B \in \mathbb{F} \implies A\cup B \in \mathbb{F}$ 
    - **Closure under Infinite Union**: $\{A_i\}_{i\in I} \subset \mathbb{F} \implies \bigcup_{i\in I}A_i \in \mathbb{F}$
    - **Top and Bottom Elements**: $\Omega, \emptyset \in \mathbb{F}$
- $\mathbb{P}_{\Omega}: 2^{\Omega} \rightarrow [0,1]$ is probability measure. 
:::

If we were so brave as to view $2^\Omega$ and $[0,1]$ as categories, we would have the following result: 

:::{prf:theorem} Measures as Functors
:label: thm-meas-as-func
The probability measure $\mathbb{P}_{\Omega}: 2^{\Omega} \rightarrow [0,1]$ is a functor on $2^{\Omega}$ viewed as a spline. 
::: 
:::{prf:proof}
:class: dropdown 
:enumerated: false
We verify functoriality. 
[FINISH]
$\mathbb{P}_{\Omega}( A \cap B ) = \mathbb{P}_{\Omega}(A) + \mathbb{P}_{\Omega}(B)$
:::