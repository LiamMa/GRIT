
# Rebuttal Draft   – GRIT

## General Update

We thank the reviewers for taking the time to review and help improve our paper. Individual comments are addressed below. We have made two minor architectural updates that are worth mentioning:

**Further Stabilizing GRIT Training** After submission, we further analyzed our GRIT model, and noticed instability of BatchNorm with larger learning rates. We have introduced a small modification to the attention computation to address the instability:

The Eq. 2 in the paper is updated from 

$$
\mathbf{e}_{i,j}' = \sigma\Big( \left(\mathbf{W}_\text{Q} \mathbf{x}_i + \mathbf{W}_\text{K}\mathbf{x}_j\right) \odot \mathbf{W}_\text{Ew}\mathbf{e}_{i,j} + \mathbf{W}_\text{Eb}\mathbf{e}_{i,j} \Big) \in \mathbb{R}^{d'}
$$

to 

$$
\mathbf{e}_{i,j}' = \sigma \Big( \text{Signed-Sqrt}\left(\mathbf{W}_\text{Q} \mathbf{x}_i + \mathbf{W}_\text{K}\mathbf{x}_j\right) \odot \mathbf{W}_\text{Ew}\mathbf{e}_{i,j} + \mathbf{W}_\text{Eb}\mathbf{e}_{i,j} \Big) \in \mathbb{R}^{d'}
$$
where $\text{Signed-Sqrt}(\mathbf{x}):= (\text{ReLU}(\mathbf{x}))^{1/2} - (\text{ReLU}(-\mathbf{x}))^{1/2}$

This $\text{Signed-Sqrt}$ function reduces the magnitude of its inputs, thus reducing the impact of large values in the attention computation; we find that this helps stabilize training. Here are updated results with this modification:


|                    | Graph MLP-Mixer  | GraphGPS        | GRIT-Old         | GRIT-New         |
|--------------------|------------------|-----------------|------------------|------------------|
| Zinc $\downarrow$            | 0.0794 $\pm$ 0.003   | 0.070 $\pm$ 0.004   | 0.066 $\pm$ 0.002    | 0.0587 $\pm$ 0.0016  |
| Mnist $\uparrow$            | 98.27 $\pm$ 0.08     | 98.051 $\pm$ 0.126  | 98.21 $\pm$ 0.154    | 98.00 $\pm$ 0.05     |
| Cifar $\uparrow$            | 70.23 $\pm$ 0.46     | 72.298 $\pm$ 0.356  | 75.63 $\pm$ 0.14     | 76.47  $\pm$ 0.88    |
| Pattern $\uparrow$          | -                | 86.685 $\pm$ 0.059  | 87.20 $\pm$ 0.08     | 87.21 $\pm$ 0.02     |
| Cluster $\uparrow$          | -                | 78.016 $\pm$ 0.180  | 79.74 $\pm$ 0.13     | 79.80 $\pm$ 0.13     |
| Peptides-func $\uparrow$    | 0.6846 $\pm$ 0.0068  | 0.6535 $\pm$ 0.0041 | 0.6665 $\pm$ 0.0106  | 0.6998 $\pm$ 0.0082  |
| Peptides-Struct $\downarrow$ | 0.2478 $\pm$ 0.0010  | 0.2500 $\pm$ 0.0012 | 0.2490 $\pm$ 0.0018  | 0.2460 $\pm$ 0.0012  |

The presented results in the rebuttal all refer to GRIT with this one small modification to the attention.


**Efficient Sparse Version of GRIT.** After the submission, we have also implemented a more efficient, sparse version of GRIT. In particular, in Sparse-GRIT, we only maintain *edge representations* (one for each edge of the input graph, of size $|E|$) instead of *node-pair representations* (of size $n^2$). This should help alleviate the concerns raised in some of the reviewer comments about scalability of our model to large graphs. In preliminary experiments, we show that Sparse-GRIT also achieves state-of-the-art performance on ZINC. 

| MAE    | GRIT-New            | Sparse-GRIT         |
|--------|---------------------|---------------------|
| Zinc   | $0.0587 \pm 0.0016$ | $0.0661 \pm 0.018$  |






## Reviewr XKJH

### W1:  The theoretical proofs do not have any added value (mostly reusing universal approximation).

While the proofs of the theorems are not necessarily very hard, we believe that they add to the literature, and can be helpful for other Graph Transformer/GNN researchers; in fact, the other reviewers seem to appreciate the results. For instance, in the biconnectivity paper (Zhang et al. 2023), shortest paths are shown to fail at distinguishing distance-regular graphs, while we show that RRWP can overcome this issue (Proposition 3.2). Moreover, while methods like GraphGPS directly incorporate message passing modules, we show that learned positional encodings initialized with RRWP can approximate various types of message passing to arbitrary accuracy (Proposition 3.1). While LayerNorm and degree information injection are used in various types of Graph Transformers, we find that one must be careful when combining them (Proposition 3.3).

### W2: Asymptotic complexity is not specified.
The asymptotic complexity of the computation of RRWP is $O(nK|\mathcal{E}|)$, where $K$ is the number of hops and $|\mathcal{E}|$ is the number of edges. Notably, RRWP is precomputed, so it only has to be computed once, and can then be used for any number of training runs.
The asymptotic complexity of GRIT is $O(|\mathcal{V}|^2)$, where $|\mathcal{V}|$ is the number of nodes, which matches other typical Graph Transformers (e.g., SAN, GraphGPS, EGT).

### W3: The experiments could be more detailed.
For the sake of fair comparison, we closely follow the experimental setup of GraphGPS (Rampášek et al. 2022) in the main experiments. Readers are referred to this paper for more details about the setup and datasets. The other details about our model, e.g., hyperparameter configuration, significance tests, and the synthetic experiments can be found in Appendix. A. We will add more details to Appendix A in the revised version.


### Q1: 1.  How much sensitive the RRWP is to the choice of the possible number of hops K? It might be useful to give a guideline (hopefully based on some theoretical justification) how to choose K based on the structure of graph.

We have run ablations on $K$ for ZINC: see this Table, which we will add to our revised paper. Notably, our method is SOTA or near SOTA for many choices of $K$; thus, RRWP is essentially not sensitive to $K$ here (except for very unreasonable choices, like $K=2$). Note that we keep every other hyperparameter besides $K$ fixed here; these hyperparameters were chosen for $K=21$, which may explain why other $K$ do a bit worse.


| MAE    | k=2              | k=7              | k=14            | k=21 (default)   | k=28             | k=42             |
|--------|------------------|------------------|-----------------|------------------|------------------|------------------|
| Zinc   | $0.147 \pm 0.006$  | $0.063 \pm 0.003$  | $0.063 \pm 0.004$  | $0.059 \pm 0.002$  | $0.063 \pm 0.003$  | $0.069 \pm 0.003$  |

> To double check with the result when new results come 



### Q2: The information from RRWP eventually reduces down to d-dimensional embedding. While it is clear that small $d$ will potentially lose some inductive bias, knowing the suitable size of $d$ with respect to the max hop $K$ would be useful.

Note that the $d$ is just the hidden dimension of the model. As such, $d$ is larger than $K$ for all of our experiments (see Table 8 in Appendix A.3); in fact, $d$ is often much larger than $K$ (e.g. $d=64$ and $K=21$ for ZINC).
It is hard to give an answer of the best $d$ w.r.t. $K$ for all tasks since it is dependent on many factors, e.g., the diameters of the graphs, the sizes of the datasets and the number of parameters of the model. We typically treat it as a hyperparameter.


### Q3:  It is less clear how to how the proposed method is effective as the sparsity of graph increases or decreases continuously.

 If “effective” refers to prediction performance, it is hard to declare a trend on the prediction performance w.r.t. the sparsity of graphs, since the practical performance is highly dependent on many factors of the datasets, e.g., the type of tasks, and other properties of the graphs.

If “effective” refers to efficiency, it is supposed not to be remarkably affected by the sparsity of the graphs, since we employ the full-attention regardless the observed edges of the graphs.
However, we have also proposed and run preliminary experiments with a sparse version of GRIT, where we only update edge representations of the original edges (as opposed to all node-pair representations). This is an efficient version of GRIT suitable to the tasks where the graphs are sparse and the efficiency is essential.



### Q4: Can you specify asymptotic complexity of your model (hopefully for other models as well)? For example, using TokenGT (Kim et al) with Laplacian embeddings and performer can get competitive performance on PCQM4Mv2 dataset with very low complexity for sparse graphs. If GRIT achieves better MAE but at sacrificing computational costs, the benefits will be limited.

The asymptotic complexity of RRWP and GRIT’s attention mechanism are $ O(nK|\mathcal{E}|)$ and $O(|\mathcal{V}|^2)$ respectively, where $K$ is the number of hops, $|\mathcal{E}|$ is the number of edges and $|\mathcal{V}|$ is the number of nodes. Our proposed method matches the asymptotic complexity of most Graph Transformers (e.g., SAN, GraphGPS, EGT).
In fact, **TokenGT's attention mechanism has a worse asymptotic complexity of $O((|\mathcal{V}|+|\mathcal{E}|)^2)$** while our method only requires $O(|\mathcal{V}|^2)$. This is because they compute a simultaneous dense attention over node and edge tokens (forming an $(n+|E|) \times (n+|E|)$ sized matrix), whereas our attention mechanism only forms an $n \times n$ matrix. **Our model reaches a better performance on PCQM4Mv2 (Table 4.) with fewer parameters and a smaller computational complexity**.

It is worth mentioning that some Graph Transformers introduce extra efficiency techniques (e.g., Performer, BigBird, Exphormer, etc) to approximate the full-attention in order to reduce the complexity, which is typically at the expense of the prediction performance of the model.
We argue that improving the performance/capacity and the scalability of Graph Transformers are actually two orthogonal directions, and this work focuses on the former direction. Further, our proposed GRIT can directly integrate with efficiency techniques such as BigBird (Zaheer et al. 2020) and Exphormer (Shirzad et al. 2023) to reach asymptotic complexity $O(|\mathcal{V}|)$, which we mostly leave for future work.  Still, as mentioned in the general comment, we implemented a sparse version of GRIT that only maintains edge representations instead of node-pair representations, and this version is still SOTA on ZINC.


More importantly, on PCQM4Mv2, TokenGT has an unconventional setup to employ Performer; they **do not train with performer attention end-to-end**. See this quote from their Appendix A.3.3: “For TokenGT (Lap) + Performer in Table 2, we load a trained model checkpoint of TokenGT (Lap), change its self-attention to FAVOR+ kernelized attention of Performer … and fine-tune it with AdamW optimizer”.
In other words, TokenGT can only enjoy a smaller complexity $O(|\mathcal{V}| + |\mathcal{E}|)$ during the evaluation stage and still requires $O((|\mathcal{V}|+|\mathcal{E}|)^2)$ during training. 


### Q5: Ablation study on ZINC could be more detailed by exploring more combinations of three proposed ingredients. Ablation study on PCQM4Mv2 will be highly exciting.

Thanks for the suggestion.
Due to the limited rebuttal time period, we have currently updated the ablation study on the choices of $K$, a version of Sparse-GRIT (see general comment), and a version without edge-updates. The other combinations of our three key design choices are still running and will be updated in the later version. 
Unfortunately, due to the limited computational resource, we are unable to conduct ablation study on PCQM4Mv2, since one trial of the experiment on PCQM4Mv2 takes a few days and using a GPU with larger memory (e.g., A100).

| MAE    | Original            | Sparse-GRIT         | GRIT-no RRWP update |
|--------|---------------------|---------------------|---------------------|
| Zinc   | $0.0587 \pm 0.0016$ | $0.0661 \pm 0.018$  | $0.0660 \pm 0.005$  |

Note that this table cannot be directly compared to the original one due to our update of hyperparameters mentioned in the general rebuttal.


### Q6: Can GRIT capture global attention only via RRWP with large $K$? Or is there any other backup to capture global attention between two nodes?
GRIT is **not** limited to capturing interactions only through RRWP with large $K$. In fact, our global attention allows us to capture global interactions between any two nodes in each layer. Distant nodes (beyond K hops) have zeros as their relative positional encodings at initialization, but can nonetheless interact in any layer with attention. Moreover, since we learn the relative positional encodings, the relative positional encodings of nodes beyond K hops are generally nonzero after the first Transformer layer.




### Q7: (Page 3, right column, line 125) Doesn’t $\boldsymbol{RW}_{ij}$ specify a probability to move from node I to node j by 1-hop rather than I hops to j?
Yes, we have the same interpretation as the reviewer, though our wording may be a bit confusing. We will reword this sentence.



- Rampášek, Ladislav, Mikhail Galkin, Vijay Prakash Dwivedi, Anh Tuan Luu, Guy Wolf, and Dominique Beaini. 2022. “Recipe for a General, Powerful, Scalable Graph Transformer.” In _Adv. Neural Inf. Process. Syst._
- Shirzad, Hamed, Ameya Velingker, Balaji Venkatachalam, Danica J. Sutherland, and Ali Kemal Sinop. 2023. “Exphormer: Scaling Graph Transformers with Expander Graphs.” _Submitted to The Eleventh International Conference on Learning Representations_.
- Zaheer, Manzil, Guru Guruganesh, Kumar Avinava Dubey, Joshua Ainslie, Chris Alberti, Santiago Ontanon, Philip Pham, et al. 2020. “Big Bird: Transformers for Longer Sequences.” In _Adv. Neural Inf. Process. Syst._ Vol. 33.




## Reviewr XmGr

### W1: Learning representation of all pairs of nodes on its own is not novel. TokenGT and EGT models include a variant of such a mechanism too, these works are generally cited in the paper but this connection is not mentioned or sufficiently contrasted to the proposed mechanism of GRIT. Additionally, also Transformer-based components of AlphaFold2 (Jumper et al., 2021) include a pair representation mechanism.

Thank you for pointing out that AlphaFold2 (Jumper et al., 2021) includes a pair representation mechanism, we will add this into our related work section.
In fact, we are not claiming that we invent "learning representation of all pairs of nodes". Instead, we employ it to enhance GRIT's ability for exploring the inductive bias from RRWP. Moreover, we provide theoretical motivation for node-pair representations in Sec. 3.1.1 and 3.2.1, which previous works do not do.

Note that TokenGT does not learn representation of all pairs of nodes, instead, it only learns the edge representations. 
EGT indeed learns representations of all pairs of nodes. However, we have notably different motivations (they propose for link prediction mentioned in Sec 3.2 of their paper) and significantly different updating mechanisms.

> Todo: reword

### Q1: Not only is the computational cost quadratic in the size of the input graph (number of nodes), also the memory complexity is quadratic. While in asymptotic complexity terms, this is true about the original GT, SAN, Graphormer, and GPS models too (except GPS with Performer global attention that has linear computational complexity), these models only incur this cost when computing the attention matrix. How necessary are the explicit pair representations you have introduced? Your theoretical results are based on the expressivity of RRWP alone. What are the trade-offs between the complexity, theoretical expressiveness, and empirical performance?



In fact, the aforementioned Graph Transformers (without approximated attention like Performer and BigBird) also require asymptotic quadratic memory complexity due to attention mechanism. 

Theoretically, our design choice is driven by the expressiveness of RRWP (Sec. 3.1.1 and Sec 3.2.1). The learning of explicit pair representations is required if we expect to explore the full expressive power of RRWP, including the ability to generalize to various existing graph PEs and message-passing propagations.
Furthermore, this can assist to infer higher-order node-pair information for distant nodes, such as edge-types/attributes between distant nodes and positional information for node-pair beyong $K$-hop.

Empirically, please find our updated ablation study on learning explicit node-pair representations.


| MAE    | Original            | GRIT no pair repr. |
|--------|---------------------|---------------------|
| Zinc   | $0.0587 \pm 0.0016$ | $0.0660 \pm 0.005$  |



### Q2: GRIT achieves worse performance on PCQM4Mv2 dataset than GPS and Graphormer, albeit theoretically GRIT should be able to at least match them. Can you please comment why? Have you managed to train any bigger/better tuned GRIT model in the meanwhile?


It may be the case that GRIT’s stronger inductive biases do not provide as much benefit on such large datasets. 
Nonetheless, we outperform most graph Transformers in the Table, besides GPS-medium and Graphormer which have $1.27$ and $3.16$ times learnable parameters of our model. Also, on small datasets where inductive biases appear to matter more (Table 1), GRIT is much better than other models including Graphormer and GraphGPS.

TODO: UPDATE BASED ON OUR FINAL RUN RESULTS


### Q3: On LRGB (Dwivedi et al., 2022) datasets GRIT achieves current SOTA on Protein-func and Protein-struct datasets, albeit the margin is rather small in the absolute terms and falls behind a recently posted Graph MLP-Mixer [1]. Have you also managed to run GRIT on the larger datasets of LRGB, such as PascalVOC-SP?

First, we would like to note that Graph MLP-Mixer is a framework employing different graph models as graph encoders (e.g. GatedGCN, GINE, original GT). Thus, Graph MLP-Mixer has an orthogonal research focus. Nonetheless, for graph-level tasks, one can easily combine Graph MLP-Mixer and GRIT by using GRIT as the base graph encoder.
Second, the latest version of our model (with modification and hyperparameter tuning mentioned in the general rebuttal section), GRIT outperforms Graph MLP-Mixer (see Table below). Further, we significantly outperform all MLP-Mixer variants on ZINC ( 0.0587 vs. 0.0745 MAE) and CIFAR-10 (76.47  vs.  72.46% accuracy).)

|                    | G-MLP-Mixer      | GRIT-Old         | GRIT-New         |
|--------------------|------------------|------------------|------------------|
| Peptides-func ↑    | 0.6846 ± 0.0068  | 0.6665 ± 0.0106  | 0.6998 ± 0.0082  |
| Peptides-Struct ↓  | 0.2478 ± 0.0010  | 0.2490 ± 0.0018  | 0.2460 ± 0.0012  |


Due to the limited time and computational resource, we have not conducted experiments on the larger datasets of LRGB.
As the future plan, we will conduct experiments on larger datasets of LRGB and potentially try the combination of Graph MLP-Mixer and GRIT.

### Q4: Missing empirical evaluation of the scalability of GRIT. Please report runtime/throughput on some of the most common tested benchmarks.

We report the runtime and memory consumption on ZINC for SAN, GraphGPS and GRIT, based on GraphGPS’s pipeline. The timing is conducted on a single V100 GPU and 20 threads of Intel(R) Xeon(R) Gold 6140 CPU @ 2.30GH.
Other models are not directly applicable in the comparison since they are based on different pipeline and/or different (graph learning) library.

| Zinc               | SAN             | GraphGPS       | GRIT           |
|--------------------|-----------------|----------------|----------------|
| PE Precompute time | 10 sec          | 11 sec         | 11 sec         |
| Memory             | 2291 MB         | 1101 MB        | 1865 MB        |
| Training time      |  57.9 sec/epoch | 24.3 sec/epoch | 29.4 sec/epoch |


> TODO: to update this

### Q5: In section 4.2 (ablation study on ZINC). A.) Have you tried Graphormer-like node degree encodings (added to node representations) instead of the degree scaler? B.) How exactly do you incorporate RRWP in the Graphormer-like attention version of the model?


 Yes, we generally found Graphormer-like node degree encodings to not perform as well, probably because our degree scalers inject degree information at each layer, as opposed to just at the input. See results on ZINC in this Table:

| GRIT       | Default          | GP-like DegEnc         |
|------------|------------------|----------------|
| ZINC (MAE $\downarrow$ ) |$0.0587 \pm 0.0016$  | $0.072 \pm 0.005$  |

The Graphormer-like Attention is defined by Eq. 6 of (Ying et al. 2021),  $A_{ij} = \frac{(h_i W_Q)(h_j W_K)^\intercal}{\sqrt{d}} + b_{\phi(v_i, v_j)}$, where $b_{\phi(v_i, v_j)}$ is a learnable scalar indexed by the shortest-path distance $\phi(v_i, v_j)$, shared across all layers.
We simply replace  $b_{\phi(v_i, v_j)}$ by $\text{MLP}(\mathbf{P}_{i,j})$ where the two-layer $\text{MLP}: \mathbb{R}^d \to \mathbb{R}$. 



### Q6: In section 4.3, it is not clear why this synthetic experiment is a regression task, instead of pair-wise classification task. What is your reasoning for choosing the regression? I find the results to be much less interpretable, particularly as the baseline MAE of a random prediction is not shown. Consider adding $R^2$ metric, or at very least MAE of a mean baseline. Additionally, models such as EGT or TokenGT should be included, as those too model (some of) pair-wise representations.

The decision of considering it as a regression task is due to the property of attention mechanism that it will generate a stochastic matrix as the attention map. It requires that the summation of each row is $1$. Therefore, it cannot be directly treated as a pair-wise classification task (multilabel classification task). Therefore, we treat it as a regression task w.r.t. the row-normalized k-hop adjacency matrix to verify the ability and flexibility of our proposed method (attention mechanism + PE scheme) to attend to specific k-hop neighbors.

Please find the attached tables for MAE and $R^2$ with the mean baseline. As suggested by the reviewer, we  have included EGT in the synthetic experiments. Note that GRIT significantly outperforms EGT in this experiment.  However, we cannot directly compare against TokenGT, as its attention mechanisms include node and edge tokens (so they are of size $n+|E| \times n+|E|$), while all other methods form attention matrices of size $n \times n$.

| $R^2$  $\uparrow$     | 1-hop            | 2-hop            | 3-hop            |
|-------------|------------------|------------------|------------------|
| MeanPooling | $0$                | $0$               | $0$                |
| RWSE        | $0$                | $0$                | $0$                |
| SAN         | $0.32 \pm 0.20$    | $0.31 \pm 0.24$   | $0.52  \pm 0.21$   |
| EGT         | $0.55 \pm 0.21$    | $0.28 \pm 0.25$   | $0.26 \pm 0.28$   |
| Graphormer  | $0.67 \pm 0.09$    | $0.77 \pm 0.08$    | $0.80 \pm 0.06$    |
| GRIT        | $0.999 \pm 0.001$  | $0.998 \pm 0.004$  | $0.961 \pm 0.035$  |


| MAE $\downarrow$         | 1-hop               | 2-hop              | 3-hop              |
|---------------|---------------------|--------------------|--------------------|
| MeanPooling |  $ 0.083 pm 0.015 $    | $0.080 \pm 0.014$    | $0.069 \pm 0.011$    |
| RWSE          |$0.083  \pm 0.015$   | $0.080  \pm 0.014$   | $0.069 \pm 0.011$    |
| SAN           |$0.044  \pm 0.011$   | $0.042 \pm 0.010$    | $0.029  \pm 0.008$  |
| EGT           | $0.030 \pm 0.012$   | $0.038 \pm 0.010$    | $0.034 \pm 0.012$    |
| Graphormer    | $ 0.043  \pm 0.010$   | $0.034 \pm 0.010$    | $0.025 \pm 0.005$    |
| GRIT          |$0.0004  \pm 0.0003$  | $0.0011 \pm 0.0004$  | $0.0067  \pm 0.004$  |




## Reviewer HYSV

### W1: The novelty of each component is not clearly stated.

We are one of the first works to propose to use uncompressed k=1,...K power of random walk matrices as relative positional encoding and demonstrate its expressive power theoretically and empirically.  

(Mialon et al. 2021) used random walk kernels that merge k-hop random walk matrices into a single one with certain fixed parameters to construct relative PE. (Li et al. 2020) utilize random walk matrices w.r.t. an anchor set of nodes to construct absolute PE (i.e., node PE).  (Zhang et al. 2023) compute resistance distances as relative PE. 
However, all of them compress the k-hop random walk matrices with certain fixed parameters to construct graph PEs, which in fact limit the expressiveness of the PE and prevent the transformer architecture from fully exploring the structural information and inductive bias of the graphs.

We also propose a new variant of attention mechanisms, which emphasizes the ability to explore the structure inductive bias provided by the relative PE. Note that even though our proposed attention mechanism is inspired by previous works (Perez et al. 2018, Brockschmidt 2020, Brody, Alon, and Yahav 2022), our proposed attention mechanism is remarkably different from them since they all only focus on node-level representations and ignore the node-pair level representations, such as relative PE. 

We also theoretically (sec 3.1.1. and sec. 3.2.1) and empirically (sec 4.3 and appx A.5) demonstrate the ability of our model to recover multiple PEs (e.g., shortest-path distance PE, random walk kernel PE, PageRank kernel PE,) and typical GNNs (e.g., message passing networks and diffusion-based GNNs). 


It is worth mentioning that previous works (Rampášek et al. 2022, He et al. 2022) often propose frameworks that they can change the architecture components and PE schemes for each task. This is in fact a drawback since the procedure of finding optimal architecture compoments and PE schems are not directly trainable and requires hyperparameter tuning.
In contrast, we have one general model that can learn well on many different tasks.

We will add these comments and make clear our contributions in the paper if accepted


- He, X., Hooi, B., Laurent, T., Perold, A., LeCun, Y., & Bresson, X. (2022). A Generalization of ViT/MLP-Mixer to Graphs. arXiv preprint arXiv:2212.13350.

- Rampášek, Ladislav, Mikhail Galkin, Vijay Prakash Dwivedi, Anh Tuan Luu, Guy Wolf, and Dominique Beaini. 2022. “Recipe for a General, Powerful, Scalable Graph Transformer.” In _Adv. Neural Inf. Process. Syst._

- Brockschmidt, Marc. 2020. “GNN-FiLM: Graph Neural Networks with Feature-Wise Linear Modulation.” In _Proceedings of the 37 Th International Conference on Machine Learning_.

- Brody, Shaked, Uri Alon, and Eran Yahav. 2022. “How Attentive Are Graph Attention Networks?” In _International Conference on Learning Representations (ICLR)_.

- Li, Pan, Yanbang Wang, Hongwei Wang, and Jure Leskovec. 2020. “Distance Encoding: Design Provably More Powerful Neural Networks for Graph Representation Learning.” In _Adv. Neural Inf. Process. Syst._

- Mialon, Grégoire, Dexiong Chen, Margot Selosse, and Julien Mairal. 2021. “GraphiT: Encoding Graph Structure in Transformers.



### W2:  Ablation studies show that the attention mechanism plays a more critical role than other components. This observation is not well emphasized in the motivation

According to our ablation studies of Table 5 on ZINC, this is true. However, we are not sure about whether this holds generally on other graph tasks. In our ablation studies, we just wanted to confirm that each design component of GRIT contributes empirically to some graph task.

###  Q1: The authors claims that the difference between RRWP and the widely used RWSE is the MLP-based updating on the position embedding. However, other works, such as LSPE [1], also update the position embedding layerwise. So, what is the difference between RRWP and LSPE?

The GNN-LSPE paper only updates node-level positional encodings (e.g. RWSE and LapPE), whereas in our paper we update node-pair-level positional encodings. Their RWSE initialization only uses the diagonals of each $(D^{-1}A)^k$ matrix, while we use the full matrix. Node-pair representations are crucial to get the full theoretical expressive power of our Propositions 3.1 and 3.2, and the full empirical expressive power in Section 4.3. We empirically show that GRIT (ours) outperforms GatedGCN-LSPE (in Table 1.).

For instance, one can show that RWSE cannot recover shortest path distances, whereas RRWP can. We will include this in the revised version: the proof is simple, and just uses the fact that RWSE assigns automorphic nodes the same embeddings.




### Q2: The empirical evaluation of theoretical results is impressive but somewhat confusing because the theoretical result states the capacity of RRWP, while the experiment is conducted with all components. So, what is the role of each component in learning K-hop relationships?

Sorry for the confusion, but our experimental setup (synthetic experiment) is basically what the reviewer wants; we will write this more clearly in the revised version. In particular, we remove degree scalers for this experiment, because we directly optimize a single attention matrix. We also remove all node-attributes and edge-attributes and use positional encoding only in these synthetic experiments; thus, our experiments measure the capacity of learnable positional encodings based on RRWP and the capacity of our attention mechanism.

