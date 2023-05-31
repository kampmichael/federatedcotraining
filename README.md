# Protecting Sensitive Data through Federated Co-Training
Implementation of A Distributed Co-Training Approach AIMHI

## Table of Contents

1. [How to Run an Experiment](#How-to-Run-an-Experiment)
2. [Appendix](#appendix)
    - [A Proof of Proposition 1](#A-Proof-of-Proposition-1)
    - [Details on Experiments](#details-on-experiments)
      - [Details on Privacy Vulnerability Experiments](#Details-on-Privacy-Vulnerability-Experiments)
      - [Datasets](#Datasets)
      - [Experimental Setup](#Experimental-Setup)
    
    
    
    
## [How to Run an Experiment](#How-to-Run-an-Experiment)
To run an experiment, you have to setup [RunExp.sh](https://github.com/kampmichael/distributedcotraining/blob/main/RunExp.sh) file with your desired parameters and then use `bash RunExp.sh` to start the experiment.
## [Appendix](#appendix)    
### [A Proof of Proposition 1](#A-Proof-of-Proposition-1)
For convenience, we restate the proposition.

**Proposition 1:** For $m\in\mathbb{N}$ clients with local datasets $D^1,\dots,D^m$ and unlabeled dataset $U$ drawn iid. from $\mathbb{D}$, let $\mathbb{A}$ be a learning algorithm that achieves a linearly increasing training accuracy $a_t$ for all labelings of $U$, i.e., there exists $c\in\mathbb{R_{+}}$ such that $a_t=1-c/t$, then there exists $t_0\in\mathbb{N}$ such that $a_t\geq 1/2$ and AIMHI with majority vote converges with probability $1-\delta$, where

$$\delta\leq|U|(4c)^{\frac{m}2}\zeta\left(\frac{m}{2},t_0+1\right)$$
and $\zeta(x,q)$ is the Hurwitz zeta function.

**Proof:**
Let $P_t$ denote the consensus label at time $t\in\mathbb{N}$. We first show that the probability $\delta_{t}$ of $P_{t}\neq P_{t-1}$ is bounded. Since the learning algorithm $\mathbb{A}$ at time $t\geq t_0$ achieves a training accuracy $a_t\geq 0.5$, the probability can be determined via the CDF of the binomial distribution, i.e.,


$$\delta_t = \mathbb{P}\left[{\exists u\in U:\sum_{i=1}^m\mathbb{1}_{h^i_t(u) = v}<\left\lfloor\frac{m}{2}\right\rfloor}\right]$$

$$=F\left(\left\lfloor\frac{m}{2}\right\rfloor-1,m,a_t\right)=\sum_{i=1}^{\left\lfloor\frac{m}{2}\right\rfloor-1}{\binom{m}{i}}a_t^i(1-a_t)^{m-i}$$

Applying the Chernoff bound and denoting by $D(\cdot\||\cdot)$ the Kullback-Leibler divergence yields


$$\delta_t \leq \exp\left(-mD\left(\frac{\left\lfloor\frac{m}{2}\right\rfloor-1}{m} \middle\|\middle\| a_t\right)^2\right)$$

$$=\exp\left(-m\left(\frac{\left\lfloor\frac{m}{2}\right\rfloor-1}{m}\log\frac{\frac{\left\lfloor\frac{m}{2}\right\rfloor-1}{m}}{a_t}+\left(1-\frac{\left\lfloor\frac{m}{2}\right\rfloor-1}{m}\right)\log\frac{1-\frac{\left\lfloor\frac{m}{2}\right\rfloor-1}{m}}{1-a_t}\right)\right)$$

$$\leq\exp\left(-m\left(\frac{\frac{m}{2}}{m}\log\frac{\frac{\frac{m}{2}}{m}}{a_t}+\left(1-\frac{\frac{m}{2}}{m}\right)\log\frac{1-\frac{\frac{m}{2}}{m}}{1-a_t}\right)\right)$$

$$=\exp\left(-m\left(\frac12\log\frac{\frac12}{a_t}+\frac12\log\frac{\frac12}{1-a_t}\right)\right)=\exp\left(-\frac{m}2\log\frac{1}{2a_t}-\frac{m}2\log\frac{1}{2(1-a_t)}\right)$$

$$=\exp\left(\frac{m}2\left(\log 2a_t + \log 2(1-a_t)\right)\right)=\left(2a_t\right)^{\frac{m}2}\left(2(1-a_t\right)^{\frac{m}2}=4^{\frac{m}2}a_t^{\frac{m}2}(1-a_t)^{\frac{m}2}\enspace.$$

The union bound over all $u\in U$ yields 
$$\delta_t\leq |U|4^{\frac{m}2}a_t^{\frac{m}2}(1-a_t)^{\frac{m}2}\enspace .$$

To show convergence, we need to show that for $t_0\in\mathbb{N}$ it holds that 
$$\sum_{t=t_0}^\infty \delta_t \leq \delta$$


for $0\leq \delta < 1$. 
Since we assume that $a_t$ grows linearly, we can write wlog. $a_t=1-c/t$ for some $c\in\mathbb{R_{+}}$ and $t\geq 2c$. With this, the sum can be written as

 $$\sum_{t=t_0}^\infty\delta_t \leq |U|\sum_{t=t_0}^\infty 4^{\frac{m}2}\left(1-\frac{c}{t}\right)^{\frac{m}2}\left(\frac{c}{t}\right)^{\frac{m}2}=|U|4^{\frac{m}2}\sum_{t=t_0}^\infty \left(\frac{\frac{t}{c}-1}{\frac{t^2}{c^2}}\right)^{\frac{m}2}$$

$$\leq|U|4^{\frac{m}2}\sum_{t=t_0}^\infty \left(\frac{\frac{t}{c}}{\frac{t^2}{c^2}}\right)^{\frac{m}2}=(4c)^{\frac{m}2}\sum_{t=t_0}^\infty \left(\frac{1}{t}\right)^{\frac{m}2}=|U|(4c)^{\frac{m}2}\zeta\left(\frac{m}{2}\right)-H_{t_0}^{\left(\frac{m}{2}\right)}$$

where $\zeta(x)$ is the Riemann zeta function and $H_n^{(x)}$ is the generalized harmonic number. Note that $H_n^{(x)}=\zeta(x)-\zeta(x,n+1)$, where $\zeta(x,q)$ is the Hurwitz zeta function, so that this expression can be simplified to

$$\sum_{t=t_0}^\infty\delta_t \leq |U|(4c)^{\frac{m}2}\zeta\left(\frac{m}{2}\right)-\zeta\left(\frac{m}{2}\right)+\zeta\left(\frac{m}{2},t_0+1\right)=|U|(4c)^{\frac{m}2}\zeta\left(\frac{m}{2},t_0+1\right)\enspace .$$


### [Details on Experiments](#details-on-experiments)
#### [Details on Privacy Vulnerability Experiments](#Details-on-Privacy-Vulnerability-Experiments)
We measure privacy vulnerability by performing membership inference attacks against AIMHI and FEDAVG.In both attacks, the attacker creates an attack model using a model it constructs from its training and test datasets. Similar to previous work[[1]](https://arxiv.org/abs/1610.05820), we assume that the training data of the attacker has a similar distribution to the training data of the client. Once the attacker has its attack model, it uses this model for membership inference. In blackbox attacks (in which the attacker does not have access to intermediate model parameters), it only uses the classification scores it receives from the target model (i.e., client's model) for membership inference. On the other hand, in whitebox attacks (in which the attacker can observe the intermediate model parameters), it can use additional information in its attack model. Since the proposed AIMHI does not reveal intermediate model parameters to any party, it is only subject to blackbox attacks. Vanilla federated learning on the other hand is subject to whitebox attacks. Each inference attack produces a membership score of a queried data point, indicating the likelihood of the data point being a member of the training set. We measure the success of membership inference as ROC AUC of these scores. The $\textbf{vulnerability (VUL)}$ of a method is the ROC AUC of membership attacks over $K$ runs over the entire training set (also called attack epochs) according to the attack model and scenario. A vulnerability of $1.0$ means that membership can be inferred with certainty, whereas $0.5$ means that deciding on membership is a random guess.

We assume the following attack model: clients are honest and the server may be semi-honest (follow the protocol execution correctly, but it may try to infer sensitive information about the clients). The main goal of a semi-honest server is to infer sensitive information about the local training data of the clients. This is a stronger attacker assumption compared to a semi-honest client since the server receives the most amount of information from the clients during the protocol, and a potential semi-honest client can only obtain indirect information about the other clients. We also assume that parties do not collude.

The attack scenario for AIMHI and DD is that the attacker can send a (forged) unlabeled dataset to the clients and observe their predictions, equivalent to one attack epoch ($K=1$); the one for FEDAVG and DP-FEDAVG is that the attacker receives model parameters and can run an arbitrary number of attacks---we use $K=500$ attack epochs.


#### [B.2:Datasets](#Datasets)
We use $3$ standard image classification datasets: CIFAR10[[2]](https://www.cs.toronto.edu/~kriz/cifar.html), FashionMNIST[[3]](https://arxiv.org/abs/1708.07747), and SVHN[[4]](https://research.google/pubs/pub37648/). We describe the datasets and our preprocessing briefly.

\textit{CIFAR10} consists of $50\,000$ training and $10\,000$ test $32\times 32$ color images in $10$ classes with equal distribution (i.e., a total of $6\,000$ images per class). Images are normalized to zero mean and unit variance.
%
\textit{FashionMNIST} consists of $60\,000$ training and $10\,000$ test $28\times 28$ grayscale images of clothing items in $10$ classes with equal distribution. Images are not normalized.
%
\textit{SVHN} (Street View House Numbers) consists of $630\,420$ $32\times 32$ color images of digits from house numbers in Google Street View, i.e., $10$ classes. The datasest is partitioned into $73\,257$ for training, $26\,032$ for testing, and $531\,131$ additional training images. In our experiments, we use only the training and testing set. Images are not normalized.

We use two standard datasets from the UCI Machine Learning repository for our experiments on collaboratively training interpretable models: WineQuality~\citep{cortez2009modeling} and BreastCancer~\citep{sudlow2015uk}. A short description of both datasets follows. 
%
\textit{WineQuality} is a tabular dataset of $6\,497$ instances of wine with $11$ features describing the wine (e.g., alcohol content, acidity, pH, and sulfur dioxide levels) and the label is a wine quality score from $0$ to $10$. We remove duplicate rows and transform the categorial type attribute to a numerical value. We then normalize all features to zero mean and unit variance.
%
\textit{BreastCancer} is a medical diagnostics tabular dataset with $569$ instances of breast cell samples with $30$ features describing cell nuclei with $2$ classes (malignant and benign). We followed the same preprocessing steps as WineQuality dataset.

Furthermore, we use $2$ medical image classification datasets, Pneumonia~\citep{kermany2018identifying}, and MRI\footnote{\url{https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection}}. 
%
\textit{Pneumonia} consists of $5\,286$ training and $624$ test chest x-rays with labels \textit{normal}, \textit{viral pneumonia}, and \textit{bacterial pneumonia}. We simplify the labels to \textit{healthy} and \textit{pneumonia} with a class imbalance of roughly $3$ pneumonia to $1$ healthy. The original images in the Pneumonia dataset do not have a fixed resolution as they are sourced from various clinical settings and different acquisition devices. We resize all images to a resolution of $224\times 224$ pixels without normalization.
%
\textit{MRI} consists of $253$ MRI brain scans with a class imbalance of approximately $1.5$ brain tumor scans to $1$ healthy scan. Out of the total $253$ images, we use $53$ images as testing set. Similar to the pneumonia dataset, the original images have no fixed resolution and are thus resized to $150\times 150$ without normalization.

#### [Experimental Setup](#Experimental-Setup)




