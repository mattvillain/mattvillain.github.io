# Topics Financial Mathematics and Artificial Intelligence

Hi, my name is Mattia Villani. 

I am a researcher in AI with applications in mathematics and finance. I did my PhD at King's under supervision of Peter McBurney and Dr. Frederik Mallmann-Trenn. Then I worked at Symbolica for a brief while, when they had just secured their series A. During and after my PhD I worked as an Applied Research Scientist at JP Morgan Chase, in the AI Research team first and then in the Quantum-Inspired Algorithms team, in Global Technology. 

I am interested in using mathematics to understand how AI works and using AI to solve mathematical problems. I am also interested in financial applications. I am collating here some of my research interests and sharing guides to how I understand topics in mathematics. 

I'd like this to be an introduction in topics that interest me. I'd like to write in a style that comes from intuitions, but also maintains a good level of formal argumentations. 

These are some of the topics I will cover:
- Mathematics for Quantitative Finance,
- Artificial Intelligence,
- Category Theory and Automated Theorem Proving. 

This is a work in progress. Content will be added over time. If you have any suggestions, please reach out to me on LinkedIn (Mattia Jacopo Villani). 

:::{admonition} Recent Research.
:class: dropdown
- [MetaTT: A global tensor-train Adapter for parameter-efficient fine-tuning](https://arxiv.org/pdf/2506.09105) (2025): you might have used LoRA before. Here we propose a new method for parameter-efficient fine-tuning. High parameter compression compared to LoRA (~10x) and comparable performance using tensor train decomposition. I helped with some engineering here. 
- [A Unified Framework for Provably Efficient Algorithms to Estimate Shapley Values](https://arxiv.org/pdf/2506.05216) (2025 - NeurIPS): there are many ways to estimate Shapley values (an important feature importance technique). Two of the main approaches are matrix-vector and least-squares estimation. We look at unifying these approaches and provide a general framework for understanding Shapley estimation. I developed the algorithmic implementation, which now works for high dimensional problems, and performed experiments. 
- [Any Deep ReLU Network is Shallow](https://arxiv.org/pdf/2306.11827) (2025 - ECAI): Nandi and I showed that you can write a network with $L$ layers as a network with $3$ layers, if you allow your weights to be in the extended reals. The proof is constructive and has some interesting implications for interpretability and pruning. 
- [Relating Piecewise Linear Kolmogorov Arnold Networks to ReLU Networks](https://arxiv.org/pdf/2503.01702) (2025 - AISTATS): Nandi, Niels and I had a look at how to relate KANs and ReLU Networks. Turns out we can write any ReLU network as a KAN and vice versa. We all developed the theory together. 
- [PICE: Polyhedral Complex Informed Counterfactual Explanations](https://ojs.aaai.org/index.php/AIES/article/view/31742&ved=2ahUKEwi3xoiOnKKTAxU7R0EAHQvCDXoQFnoECBwQAQ&usg=AOvVaw2Gk282nXz_s0d3AaOlYow3) (2025 - AIES in AAAI): we can get exactly minimal counterfactuals for ReLU networks, with sufficient compute. This was my internship project. 
- [Trading-off accuracy and communication cost in federated learning](https://arxiv.org/pdf/2503.14246) (2025 - AAMAS): using training by sampling, we can get incredibly cheap communication costs in federated learning. There's also a nice connection to Zonotope geometry. 
- [Graph Convolutional Neural Networks as Parametric CoKleisli morphisms](https://arxiv.org/abs/2212.00542) (2022 - SYCO-10 workshop): Bruno and I sat down to look at GCNs via category theory. This paper was the result of a weekend together in London. The paper just describes weight sharing in GCNs. [Bruno presented it nicely here](https://www.cl.cam.ac.uk/events/syco/10/slides/gavranovic.pdf). 
- [Feature Importance for Time Series Data: Improving KernelSHAP](https://arxiv.org/pdf/2210.02176)(2022 - ICAIF workshop): using Shapley value as a feature importance techniques relies on the assumption that every feature is a player in a game. If our model is a time series model, this isn't quite right. We propose a method for computing Shapley for time series. This was my internship project. 
:::

:::{admonition} Past Projects.
:class: dropdown
- [United Italian Societies](https://uniteditaliansocieties.com) (2017-2024): I helped Umberto set up this non-profit organisation, back in 2022. Before that, I managed a precursor project called Italian Societies Student Network. Within the UIS I founded the [Research Centre](https://www.linkedin.com/showcase/uis-research-centre/). We helped Italian students in the UK connect, orgnaise educational events and provided support. 
- [The Red Flower Factory](https://theredflowerfactory.com) (2020): I helped set up this spin-off start-up from Sevendots - a CPG marketing consultancy. 
:::