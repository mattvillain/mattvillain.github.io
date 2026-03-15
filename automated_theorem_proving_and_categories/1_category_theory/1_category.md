# Defining Categories

So what is a category? A category looks like a transitive, directed multi-graph. However, whenever you start talking about graphs you must start with a tuple of sets $(V,E)$. We don't want to use sets - we can build foundations of mathematics without having to resort to set theory. 

:::{prf:definition} Category
:label: def-category

A category $C$ is a collection of objects $ob(C)$ and a collection of morphisms $mor(C)$, together with the following maps:
- $dom, cod: mor(C) \rightarrow ob(C)$ selects the domain and codomain for each morphism, 
- $id: ob(C) \rightarrow mor(C)$ assigns to each object an identity morphism,
- $\circ: mor(C) \times mor(C) \rightarrow mor(C)$ is a composite (generally partial) map. 
:::

A category describes how you can compose certain families of maps. We can have the following examples of categories: 
- $\textbf{Vect}_K$ the category of vector spaces on a field $K$ ($\mathbb{R}$ is a popular choice), and bilinear maps. 
- $\textbf{Smooth}$ the category of smooth manifolds and diffeomorphisms. 
- $\textbf{Group}$ the category of group and group homomorphisms.
- $\textbf{Markov}$ the category of probability spaces and markov kernels between them. 
- $\textbf{Top}$ the category of topological spaces and continuous functions between them. 
- $\textbf{Func}$ the category of functors and natural transformations between them (this will make sense later). 

