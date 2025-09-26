Reviewer L2j8

We thank the Reviewer for his/her careful reading of the paper and for the positive feedback and suggestions received. We believe we completely address below all the comments from the Reviewer, and will be happy to answer any further question the Reviewer could have during the authors/reviewers discussion period.

*Additional experimental results with more diverse datasets*

The experimental results in the paper show results using 8 publicly available real-world datasets commonly used to compare boosting methods. Five of them are available in UCI repository and the rest in Kaggle website. The instance dimensionality and label proportions in the datasets vary significantly among the datasets. Following the comment from the Reviewer, we have added additional results with other types of datasets that have a larger number of samples. The following table shows the additional results obtained using 5,000 samples from the datasets Susy, Higgs, and Forestcov that can be found in UCI repository.

|           | Dataset   | RobustB  | AdaB     | LPB      | LogitB   | GentleB  | BrownB   | GBDT     | XGB$-$Q    | RMB      | Minimax  |
|-----------|-----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|
|           | Susy      | 24±1.7 | 24±1.8 | 30±2.2 | 24±2.0 | 25±1.7 | 23±1.7 | 25±1.8 | 24±1.6 | 23±2.0 | 23±0.3 |
| Noiseless | Higgs     | 34±1.8 | 33±2.0 | 38±3.1 | 33±1.9 | 34±2.1 | 34±1.9 | 37±2.0 | 37±2.4 | 35±2.1 | 33±0.3 |
|           | Forestcov | 20±1.8 | 20±1.8 | 27±1.7 | 17±1.1 | 20±1.7 | 20±1.5 | 26±2.0 | 33±1.7 | 22±1.6 | 22±0.2 |
| | | | | | | | | | | | |
|           | Susy      | 26±2.2 | 24±1.8 | 35±3.4 | 27±1.9 | 28±1.8 | 24±1.8 | 26±1.8 | 24±1.8 | 23±1.9 | 28±0.4 |
| $P_{\text{noise}} = 10%$     | Higgs     | 35±2.1 | 35±1.9 | 41±1.4 | 36±2.1 | 37±2.3 | 35±2.2 | 39±2.2 | 38±2.4 | 35±2.4 | 36±0.5 |
|           | Forestcov | 22±1.7 | 22±1.7 | 34±1.9 | 22±1.2 | 25±2.0 | 23±1.6 | 27±2.1 | 33±2.2 | 23±1.7 | 28±0.4 |
| | | | | | | | | | | | |
|           | Susy      | 27±2.0 | 26±2.0 | 38±2.2 | 30±2.1 | 32±2.0 | 25±1.9 | 29±1.6 | 25±1.7 | 24±2.0 | 32±0.5 |
|  $P_{\text{noise}} = 20%$       | Higgs     | 37±2.3 | 37±2.0 | 46±3.3 | 38±2.3 | 39±2.4 | 37±2.3 | 41±2.9 | 39±2.7 | 36±2.2 | 39±0.6 |
|           | Forestcov | 24±1.8 | 24±1.8 | 38±2.3 | 25±1.5 | 29±1.9 | 25±1.9 | 32±2.1 | 34±2.7 | 23±2.0 | 32±0.4 |
| | | | | | | | | | | | |
|           | Susy      | 32±2.0 | 32±2.1 | 43±2.7 | 34±2.0 | 35±2.0 | 31±3.3 | 34±1.6 | 33±2.9 | 32±2.3 | 28±0.5 |
| Adversarial  $P_{\text{noise}} = 10%$   | Higgs     | 39±2.0 | 39±2.0 | 48±2.8 | 41±2.0 | 42±2.3 | 38±3.4 | 44±1.2 | 38±1.7 | 38±2.4 | 40±0.5 |
|           | Forestcov | 28±1.9 | 28±2.0 | 40±2.2 | 28±2.1 | 29±2.3 | 28±2.0 | 27±1.7 | 28±2.3 | 28±2.1 | 31±0.5 |
| | | | | | | | | | | | |
|           | Susy      | 40±2.0 | 40±1.9 | 49±2.8 | 43±1.9 | 44±2.2 | 40±1.5 | 41±2.3 | 39±2.0 | 38±2.5 | 33±0.5 |
|  Adversarial $P_{\text{noise}} = 20%$       | Higgs     | 49±2.1 | 50±2.2 | 49±3.4 | 49±2.7 | 49±2.4 | 49±2.8 | 49±0.7 | 49±4.1 | 48±3.3 | 46±0.6 |
|           | Forestcov | 39±2.5 | 39±2.4 | 47±3.8 | 38±2.2 | 40±2.2 | 39±2.1 | 38±1.4 | 36±2.7 | 36±2.6 | 36±0.7 |

The table above with additional results shows similar behavior as that in the experimental results of the submitted manuscript. The proposed methods achieve adequate performance without noise together with improved robustness with noisy labels.

*Generality of the label noise considered in the paper and one-sided label noise*

We would like to emphasize that the label noise considered in the paper is totally general (lines 78-79), and includes the one-sided label noise in [1] [2]. Specifically, any label noise is described by a function $\rho_y(x)$ that quantifies the probability with which the label of an instance $x$ is flipped from $y$ to $-y$. In the case of one-side label noise, the function $\rho_y(x)$ satisfies that $\rho_+1(x)=0$ and $0<=\rho_-1(x)<=1$ for all $x$, for cases where only the negative samples are affected by noise; respectively, $0<=\rho_+1(x)<=1$ and $\rho_-1(x)=0$ for all $x$, for cases where only the positive samples are affected by noise. The results in the paper consider arbitrary functions $\rho_y(x)$ so that they are valid for any type of label noise. We thank the Reviewer for pointing out the type of noise “one-sided label noise” that will be described in the final version of the paper updating the lines 78-86 with the corresponding references.

We agree with the Reviewer that incorporating one-sided label noise into the experimental results can further validate the methods proposed. So that we have carried out other additional results in which we use one-sided noise, both uniform and adversarial (only one class affected by noise). The following table shows these additional results with one-sided label noise in experimental results as those shown in Figure 2 of the submitted manuscript.


|          |          | Method | 3%       | 6%       | 9%       | 12%       | 15%      | 18%       | 21%      | 24%      | 27%      | 30%      |
|----------|----------|--------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|
| Uniform|    Diabetes       | AdaB   | 27±4.5 | 27±5.5 | 29±5.2 | 30±5.2 | 31±5.2 | 32±5.6 | 33±5.4 | 34±5.2 | 35±5.5 | 38±5.6 |
| noise | | RMB    | 26±4.8 | 26±5.0 | 27±4.8 | 27±5.3 | 27±4.4 | 29±4.9 | 29±5.4 | 30±5.7 | 32±5.6 | 35±5.3 | 
| | | | | | | | | | | | |
| Adversarial |    Diabetes       | AdaB   | 32±5.8 | 34±4.7 | 34±4.5 | 35±4.7 | 37±4.2 | 44±4.3 | 45±4.3 | 46±4.2 | 47±4.2 | 48±4.2 |
| noise | | RMB    | 26±4.5 | 26±5.7 | 27±4.9 | 32±6.5 | 35±4.0 | 36±4.7 | 36±5.6 | 38±5.2 | 40±4.3 | 43±4.8 |
| | | | | | | | | | | | |
| Uniform|  Raisin          | AdaB   | 15±3.5 | 16±3.7 | 16±3.7 | 17±4.1 | 19±3.9 | 20±3.8 | 20±4.3 | 22±4.5 | 23±4.2 | 25±3.7 |
| noise |          | RMB    | 15±3.4 | 15±3.6 | 15±4.0 | 15±3.9 | 15±3.6 | 16±2.4 | 17±4.1 | 17±4.0 | 18±4.3 | 19±4.5 |
| | | | | | | | | | | | |
| Adversarial |    Raisin       | AdaB   | 17±3.5 | 18±3.7 | 21± 3.9 | 22±3.7 | 24±3.7 | 24±3.5 | 26±3.9 | 31±4.1 | 31±3.9 | 31±3.7 |
| noise | | RMB    | 14±3.9 | 17±2.7 | 17±2.6 | 18±3.9 | 19±3.4 | 19±2.7 | 20±5.0 | 23±4.7 | 24±3.7 | 27±4.8 |

The results above shows that the proposed techniques are also robust to one-side label noise. 

Reviewer TwM6

We thank the Reviewer for his/her appreciation of the theoretically well-grounded algorithms and the solid theoretical contribution. We also thank the Reviewer for the suggestions and comments provided that are addressed below. We will be happy to answer any further question the Reviewer could have during the authors/reviewers discussion period.

*Experimental comparison with the methods analyzed in reference [14]*

As described by the Reviewer, the results in reference [14] show robustness results that go beyond the common symmetric and uniform noise to symmetric. Specifically, such work shows that certain potential functions (e.g., sigmoid) are also robust to symmetric and non-uniform label noise (in cases where the Bayes risk is zero). The submitted paper describes the results in [14] as one of the few works that go beyond the symmetric and uniform noise. However, the methods described in [14] are not directly comparable experimentally with the methods proposed: the results in [14] are not oriented to boosting methods and the potential functions analyzed are non-convex. Specifically, the classification rules considered in [14] are linear rules and quadratic-kernel rules, not ensembles of base-rules. In addition, the reference [14] utilizes dedicated optimization algorithms to dead with the non-convexity of the potential function, which have an unclear generalization in boosting settings. Nevertheless, over the past few days, we have used the XGBoost library to implement boosting methods with the potential functions proposed in [14]. This library allows the use of customized potential/loss functions. However, the performance obtained was significantly worse than that of all the other methods. We guess this is due to the non-convexity of the potential functions.

*Additional experimental results with larger datasets*

The specific learning algorithm proposed in Section 5 of the paper to solve the minimax problem (4) is based on column generation approach for linear optimization, similarly as other methods such as LPBoost. A known limitation of approaches based on column generation is that their complexity increases with the number of samples faster than other boosting methods such as AdaBoost (see discussion on lines 279-306 and running times in Figure 5 comparing AdaBoost, LPBoost, and the proposed RMBoost). Notice also that approaches based on column generation have not been implemented in the reference [B]. For instance, the size of the optimization problems solved by the column generation methods increases linearly with the number of samples. Therefore, the usage of such approaches with large scale datasets with millions of samples require specialized optimization methods such as parallel solvers, dual decompositions or core-set selection, which go beyond the scope of the submitted paper.  In the final version of the paper, we will extend the description of this topic in the paragraph about computational complexity in relation with the reference, and state this limitation in the new section for limitations.

Nevertheless, we agree with the Reviewer that it is of interest to include additional experiments with larger datasets than those used in the submitted manuscript (up to 1k samples). Hence, we have carried out new experimental results with the datasets suggested by the Reviewer, where the limitations of the column generation methods have been addressed by using a subset of the datasets composed by 5,000 samples. The following table shows the results obtained for different types of noise.

|           | Dataset   | RobustB  | AdaB     | LPB      | LogitB   | GentleB  | BrownB   | GBDT     | XGB$-$Q    | RMB      | Minimax  |
|-----------|-----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|
|           | Susy      | 24±1.7 | 24±1.8 | 30±2.2 | 24±2.0 | 25±1.7 | 23±1.7 | 25±1.8 | 24±1.6 | 23±2.0 | 23±0.3 |
| Noiseless | Higgs     | 34±1.8 | 33±2.0 | 38±3.1 | 33±1.9 | 34±2.1 | 34±1.9 | 37±2.0 | 37±2.4 | 35±2.1 | 33±0.3 |
|           | Forestcov | 20±1.8 | 20±1.8 | 27±1.7 | 17±1.1 | 20±1.7 | 20±1.5 | 26±2.0 | 33±1.7 | 22±1.6 | 22±0.2 |
| | | | | | | | | | | | |
|           | Susy      | 26±2.2 | 24±1.8 | 35±3.4 | 27±1.9 | 28±1.8 | 24±1.8 | 26±1.8 | 24±1.8 | 23±1.9 | 28±0.4 |
| $P_{\text{noise}} = 10%$     | Higgs     | 35±2.1 | 35±1.9 | 41±1.4 | 36±2.1 | 37±2.3 | 35±2.2 | 39±2.2 | 38±2.4 | 35±2.4 | 36±0.5 |
|           | Forestcov | 22±1.7 | 22±1.7 | 34±1.9 | 22±1.2 | 25±2.0 | 23±1.6 | 27±2.1 | 33±2.2 | 23±1.7 | 28±0.4 |
| | | | | | | | | | | | |
|           | Susy      | 27±2.0 | 26±2.0 | 38±2.2 | 30±2.1 | 32±2.0 | 25±1.9 | 29±1.6 | 25±1.7 | 24±2.0 | 32±0.5 |
|  $P_{\text{noise}} = 20%$       | Higgs     | 37±2.3 | 37±2.0 | 46±3.3 | 38±2.3 | 39±2.4 | 37±2.3 | 41±2.9 | 39±2.7 | 36±2.2 | 39±0.6 |
|           | Forestcov | 24±1.8 | 24±1.8 | 38±2.3 | 25±1.5 | 29±1.9 | 25±1.9 | 32±2.1 | 34±2.7 | 23±2.0 | 32±0.4 |
| | | | | | | | | | | | |
|           | Susy      | 32±2.0 | 32±2.1 | 43±2.7 | 34±2.0 | 35±2.0 | 31±3.3 | 34±1.6 | 33±2.9 | 32±2.3 | 28±0.5 |
| Adversarial  $P_{\text{noise}} = 10%$   | Higgs     | 39±2.0 | 39±2.0 | 48±2.8 | 41±2.0 | 42±2.3 | 38±3.4 | 44±1.2 | 38±1.7 | 38±2.4 | 40±0.5 |
|           | Forestcov | 28±1.9 | 28±2.0 | 40±2.2 | 28±2.1 | 29±2.3 | 28±2.0 | 27±1.7 | 28±2.3 | 28±2.1 | 31±0.5 |
| | | | | | | | | | | | |
|           | Susy      | 40±2.0 | 40±1.9 | 49±2.8 | 43±1.9 | 44±2.2 | 40±1.5 | 41±2.3 | 39±2.0 | 38±2.5 | 33±0.5 |
|  Adversarial $P_{\text{noise}} = 20%$       | Higgs     | 49±2.1 | 50±2.2 | 49±3.4 | 49±2.7 | 49±2.4 | 49±2.8 | 49±0.7 | 49±4.1 | 48±3.3 | 46±0.6 |
|           | Forestcov | 39±2.5 | 39±2.4 | 47±3.8 | 38±2.2 | 40±2.2 | 39±2.1 | 38±1.4 | 36±2.7 | 36±2.6 | 36±0.7 |




The additional results show similar behavior as that in the experimental results of the submitted manuscript. The proposed methods achieve adequate performance without noise together with improved robustness with noisy labels.

[B] Chen et. al., Robustness Verification of Tree-based Models, in NeurIPS 2019.

*Specific section dedicated to the limitations of the methods proposed*

We thank the Reviewer for the suggestion provided about a section dedicated to the methods’ limitations in terms of training complexity. In the submitted manuscript, the limitations are described on lines 279-306 and lines 658-668. As suggested by the Reviewer, the new section for limitations will state the additional computational time required by the column generation method in comparison with other existing approaches, and the related limitation in the usage of datasets with millions of samples (the final version of the paper will include the new experimental results shown above).

Minor comments

We thank the Reviewer for pointing few typos and suggestions to improve the writing that will be addressed in the final version of the paper. Regarding the usage of $\phi$ or $\phi'$ in line 107, the correct usage is $\phi'(0)<0$ since the condition required for a potential function is that the derivative at zero is negative, that is, the potential function should be decreasing at zero since positive margins should be preferred to negative margins (see condition 2 in Definition 1 of reference [10]).

Reviewer 5GJn

We thank the Reviewer for the suggestions provided regarding notations and more intuitive descriptions. We consider the explanations provided below can help the Reviewer to get a more clear idea of the results presented and the relation with the related work. In case there are still some notational barriers limiting the results clarity, we would appreciate if the Reviewer let us know what specific notations are unclear so that we could further elaborate in the updated manuscript. 

*Notation clarification and intuition for uncertainty sets and Theorem 1*

We plan to use the extra page allowed in the final version of the paper to further clarify the notation used with a more intuitive description. Regarding base-rules, the theoretical sections of the paper consider general families of base-rules  (any family of bounded functions from $\mathcal{X}$ to $[-1,1]$). In some parts of the main text as lines 246-248, 321-324, and in the experimental results, the family of base-rules is given by decision trees, which are the common type of base-rules used in boosting methods.

The uncertainty set in (5) is composed by all the probability distributions over instance-label pairs for which expectations of the base-rules are near their empirical averages over the training samples. Specifically, the expectations used to define the uncertainty set are correlations between labels and base-rules predictions $(y h(x))$. The rationale for the usage of such uncertainty set is to have a small set that can contain the true underlying distribution of samples with high probability. The uncertainty set in (5) is small because the family of base rules is often very large so that (5) is given by a large number of constraints. The uncertainty set in (5) contains the true underlying distribution with high probability because base-rules are simple functions for which uniform convergence of sample averages is ensured.

Theorem 1 shows that the minimax problem in (4) is equivalent to a tractable convex optimization problem (linear optimization with L1-regularization). This result enables to address (4) in practice by solving the tractable problem (6). The main idea behind the proof of Theorem 1 is substituting the inner maximization in (4) by its Fenchel dual, so that the minimax problem becomes a convex optimization problem.

*More intuitive formulation of Equations (5) and (7)*

Regarding equation (5) for the uncertainty set, other way to formulate the same uncertainty set is as follows. Let $\mathcal{H}=\{h_1,h_2,…h_T}$ be the set of all base-rules, e.g., all the decision trees with a bounded number of nodes given by components of the instances in the training set. The uncertainty set in (4) is

$$\mathcal{U}=\{prob. dist. p over X\times Y such that |\mathbb{E}_p y h_j(x)-\frac{1}{n}\sum_{i=1}^n y_i h_j(x_i)|\leq \lambda, for all j=1,2,…T\}$$

that is, $\mathcal{U}$ is given by the $2T$ linear constraints 

\frac{1}{n}\sum_{i=1}^n y_i h_j(x_i)-\lambda<= \sum_{x,y} p(x,y) yh_j(x)<= \frac{1}{n}\sum_{i=1}^n y_i h_j(x_i)+\lambda

The Lagrange (Fenchel) multipliers of such constraints correspond to the parameters \mu in Theorem 1.

Regarding equation (7) for the minimax classification rule, other way to formulate the equation defining the classification rule is as follows. Let \mu^*=\mu^*_1,\mu^*_2,…\mu^*_T be the solution of optimization problem (7).

A solution of the minimax problem in (4) is given by

h_\mu^*(y|x)=y \sum_{j=1}^T h_j(x)\mu^*_j +1/2

that is, the minimax classification rule is given by a linear combination of the base-rules \mathcal{H}=\{h_1,h_2,…h_T} with coefficients given by a solution of the optimization problem (6). For instances $x$ for which the combination of base-rules $\sum_{j=1}^T h_j(x)\mu^*_j$ is positive, the probability $h_\mu^*(y=+1|x) $ is larger than 1/2 and hence larger than $h_\mu^*(y=-1|x)$, so that the deterministic classifier predicts label +1, (otherwise the classifier predicts label -1). Therefore, the determinist minimax classifier is given by

h_\mu^*(x)=sign(\sum_{j=1}^T h_j(x)\mu^*_j) 

as shown in equation (8).
