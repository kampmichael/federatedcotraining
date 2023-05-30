# Protecting Sensitive Data through Federated Co-Training
Implementation of A Distributed Co-Training Approach AIMHI

## Table of Contents

1. [How to Run an Experiment](#howtorun)
2. [Appendix](#appendix)
    - [A: A Proof of Proposition 1](#proofofproposition)
    - [B: Details on Experiments](#b-details-on-experiments)
    
    
    
    
## [How to Run an Experiment](#howtorun)      
## [Appendix](#appendix)    
### [A: A Proof of Proposition 1](#proofofproposition)
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


























### [B: Details on Experiments](#b-details-on-experiments)



