---
title: On Moment Matching
date: 2025-07-02
author: ewerlopes
description: Understanding the role of the normalization coefficient in the Exponential Family and its connection to Moment Matching
language: en
math: true
categories:
  - bayesian
  - inference
  - divergence-measures
tags:
  - posterior-approximation
  - kl-divergence
  - exponential-family
draft: true
---

I am not sure about you, but I certainly lost track of how many times I re-read this passage from Chris Bishop's classical Pattern Recognition & Machine Learning book:

>"More generally, it is straightforward to obtain the required expectations_ for any member of the exponential family, provided it can be normalized, because the expected statistics can be related to the derivatives of the normalization coefficient, as given by (2.226)" (Chapter 10, p. 508).

Wow! Every time I see a "it is straightforward" (similarly, "gentle introduction") in an statistics or respectable ML book my first reaction is to brace for impact. Bishop is no exception. As I mentioned, the passage was certainly no stranger to me, and yet I felt like it quite did not cling to my memory in any substantial way. The "obtain the required expectations", the "provided it can be normalized" and the "derivatives of the normalization coefficient" where essentially water and oil. I then decided to invest a substantial chunk of my not so efficiently-spent time, in between the nightly cries of my little daughter, to settle the matter. Slowly but surely, I commenced the struggle.

## Preliminaries

As we will soon see, the key to grasp what Bishop's is saying is *Moment Matching*, a technique essential for doing approximate inference in bayesian models where the distributions used to approximate the posterior are constrained to the exponential family. **Expectation Propagation** (Minka, 2001), a subject approached in the chapter that passage was taken from, relies entirely on the technique.

To make sense of it all, we need to recall the core concepts about Exponential Family. In short, a probability distribution is said to belong to the Exponential Family of distributions if it can be written as

$$
\begin{aligned}
p(\boldsymbol{x}|\boldsymbol{\theta}) &= \frac{1}{Z(\boldsymbol{\theta})}h(\boldsymbol{x})exp\{\boldsymbol{\theta}^{T}\boldsymbol{\phi}(\boldsymbol{x})\} \\
&= h(\boldsymbol{x})exp\{\boldsymbol{\theta}^{T}\boldsymbol{\phi}(\boldsymbol{x}) - A(\boldsymbol{\theta})\}
\end{aligned}
$$
, where

- $h(\boldsymbol{x})$: is the is the underlying function measure, capturing the distribution's volume.
- $Z(\boldsymbol{\theta})$: is the partition function (you may safely call it the normalizing function) that ensure the distribution sums up to one, being defined as $\int h(\boldsymbol{x})exp\{\boldsymbol{\theta}^{T}\boldsymbol{\phi}(\boldsymbol{x})\}d\boldsymbol{x}$ .
- $\boldsymbol{\theta}$: is the natural parameters and $\boldsymbol{\theta} \in \mathbb{R}^d$.
- $\boldsymbol{\phi}(\boldsymbol{x})$: is the sufficient statistics.
- $A(\theta)$: is the log partition function given as $\ln{Z(\theta)}$.

## Differentiating the log partition function

Undeniably, the partition function takes a legendary role in model inference. We see it everywhere one needs to make prediction or model comparison. In his passage, Bishop talks about "expectations", "derivatives" and "normalization coefficients". Assuming a member of the Exponential Family (to boost your intuition, think about a Gaussian distribution), what would we uncover by taking the gradient of the partition function?

Let's start from the fact that exponential family members integrate to $1$ (Recall from the previous section that the partition function ensures that the whole distribution sums up to one).

$$
\begin{aligned}
\int p(\boldsymbol{x}|\boldsymbol{\theta}) d\boldsymbol{x} &=  \int h(\boldsymbol{x})exp\{\boldsymbol{\theta}^{T}\boldsymbol{\phi}(\boldsymbol{x}) - A(\boldsymbol{\theta})\}d\boldsymbol{x} = 1 \\
\end{aligned}
$$
Taking the derivative on both sides
$$
\begin{aligned}
\nabla \left( \int p(\boldsymbol{x}|\boldsymbol{\theta}) d\boldsymbol{x} \right) &= \nabla \left( \int h(\boldsymbol{x})exp\{\boldsymbol{\theta}^{T}\boldsymbol{\phi}(\boldsymbol{x}) - A(\boldsymbol{\theta})\}d\boldsymbol{x} \right) = 0
\end{aligned}
$$

Then, by relying on a special case of the [[Leibniz Rule]], we get to:

$$
\begin{aligned}
\nabla \int p(\boldsymbol{x}|\boldsymbol{\theta}) d\boldsymbol{x} &= \nabla \left( \int h(\boldsymbol{x})exp\{\boldsymbol{\theta}^{T}\boldsymbol{\phi}(\boldsymbol{x}) - A(\boldsymbol{\theta})\}d\boldsymbol{x} \right)\\
&= \int \nabla \Biggl( h(\boldsymbol{x})exp \Biggl\{\boldsymbol{\theta}^{T}\boldsymbol{\phi}(\boldsymbol{x}) - A(\boldsymbol{\theta})\Biggl\} \Biggl)d\boldsymbol{x} \\
&= \int \underbrace{h(\boldsymbol{x})exp \Biggl\{\boldsymbol{\theta}^{T}\boldsymbol{\phi}(\boldsymbol{x})- A(\boldsymbol{\theta}) \Biggl\} }_{p(\boldsymbol{x}|\boldsymbol{\theta})} \Biggl(\boldsymbol{\phi}(\boldsymbol{x}) - \nabla A(\boldsymbol{\theta})\Biggl) d\boldsymbol{x} \\
&= \int p(\boldsymbol{x}|\boldsymbol{\theta})\boldsymbol{\phi}(\boldsymbol{x})d\boldsymbol{x} - \int p(\boldsymbol{x}|\boldsymbol{\theta}) \nabla A(\boldsymbol{\theta}) d\boldsymbol{x} \\
&= \langle \boldsymbol{\phi}(\boldsymbol{x}) \rangle_{p(\boldsymbol{x}|\boldsymbol{\theta})} - \nabla A(\boldsymbol{\theta})
\cancelto{1}{\int p(\boldsymbol{x}|\boldsymbol{\theta})d\boldsymbol{x}} \\
&= 0
\end{aligned}
$$

and therefore

$$
\nabla A(\boldsymbol{\theta}) = \Big \langle \boldsymbol{\phi}(\boldsymbol{x}) \Big\rangle_{p(\boldsymbol{x}|\boldsymbol{\theta})}
$$

, where we write $\langle g(x) \rangle _{p(x)}$ as a shorthand notation for the expectation of $g(x)$ under $p(x)$as in $\int g(x)p(x)dx$. This results show us that we can compute moments of the exponential distribution through differentiation of the log partition function (recall that the expected value is the first raw moment of a distribution). Why is this good? Well, for starters, differentiation is typically easier than integration and, additionally, as we will see, this will turn out be quite a important result for the minimization of the [[KL-divergence]].

> "Thus, provided we can normalize a distribution from the exponential family, we can always find its moments by simple differentiation". (Chapter 2, p. 116)

That is it! End of mystery. The role of the normalization coefficient in the exponential distribution is such that it allow us to compute expectations over the sufficient statistics. The "because the expected statistics can be related to the derivatives of the normalization coefficient" is much clearer now.

As an exercise you can compute hight-order derivates and discover that the covariance of $\boldsymbol{\phi}(\boldsymbol{x})$ can be expressed in terms of the second derivative of $A(\boldsymbol{\theta})$. Higher order moments will similarly relate to higher order derivatives.

But, wait! How does it all connect to moment matching? Let see this next.

# Why "Moment matching"?

The KL-divergence is central to posterior approximation techniques like Variational Inference and [[Expectation Propagation]]. In the later, the quantity defines a criteria that guides the selection of the best approximating member from a family of distribution. It is when the exponential family is used as the approximation family in the minimization of the KL that an incredible result emerges and the term "moment matching" becomes self-explainable. The following theorem captures it well.

> **THEOREM 1:** For a distribution $p(\boldsymbol{x}|\boldsymbol{\theta})$, the distribution $q(\boldsymbol{x}|\boldsymbol{\theta^*})$ which minimizes the KL-divergence, $KL(p(\boldsymbol{x}|\boldsymbol{\theta})||q(\boldsymbol{x}|\boldsymbol{\theta^*}))$, over the exponential family with natural statistics $\boldsymbol{\phi}$ is implicitly given by $ \langle \boldsymbol{\phi}(\boldsymbol{x}) \rangle_{q(\boldsymbol{x}|\boldsymbol{\theta^*})} = \langle \boldsymbol{\phi}(\boldsymbol{x}) \rangle_{p(\boldsymbol{x}|\boldsymbol{\theta})}$

Now, if you're asking yourself where the "self-explainable" in the previous paragraph is, just observe that the term "moment" refers to the expectations in the theorem. After all, an expectation is the first raw moment of a distribution. The "matching" is a reference to the equality in the theorem above. Next, we sketch a proof (you can read more about the proof [here](https://www.herbrich.me/papers/KL.pdf)).

We start by considering the KL-divergence as a function $f$ of the parameters $\theta$. We will also simplify the distribution notation by removing the reference to its latent parameters.

$$
\begin{aligned}
f(\boldsymbol{\theta}) &= KL(p(\boldsymbol{x})||q(\boldsymbol{x}))
= \Biggl \langle \ln \Biggl( \frac{p(\boldsymbol{x})}{q(\boldsymbol{x})}\Biggl) \Biggl \rangle_{p(\boldsymbol{x})}  \\
&= \int p(\boldsymbol{x}) \ln p(\boldsymbol{x}) d\boldsymbol{x} - \int p(\boldsymbol{x}) \ln q(\boldsymbol{x}) d\boldsymbol{x} \\
&= \int p(\boldsymbol{x}) \ln p(\boldsymbol{x}) d\boldsymbol{x} - \int p(\boldsymbol{x}) \ln \Biggl( h(\boldsymbol{x}) exp \Biggl\{ \boldsymbol{\theta^T}\boldsymbol{\phi}(\boldsymbol{x}) - A(\boldsymbol{\theta}) \Biggl\} \Biggl) d\boldsymbol{x} \\
&= \int p(\boldsymbol{x}) \ln p(\boldsymbol{x}) d\boldsymbol{x} - \int p(\boldsymbol{x}) \ln h(\boldsymbol{x}) d\boldsymbol{x} - \int p(\boldsymbol{x}) \boldsymbol{\theta^T}\boldsymbol{\phi}(\boldsymbol{x}) d\boldsymbol{x} + \int p(\boldsymbol{x}) A(\boldsymbol{\theta}) d\boldsymbol{x}
\end{aligned}
$$
We then need to compute the gradient w.r.t. $\boldsymbol{\theta}$ and set it to zero

$$
\begin{aligned}
\nabla_{\boldsymbol{\theta}} f(\boldsymbol{\theta}) &= \nabla_{\boldsymbol{\theta}} \Biggl( \langle \ln(p(\boldsymbol{x})\rangle_{p(\boldsymbol{x})} - \langle \ln(h(\boldsymbol{x})\rangle_{p(\boldsymbol{x})} - \langle \boldsymbol{\theta^T}\boldsymbol{\phi}(\boldsymbol{x})\rangle_{p(\boldsymbol{x})} + A(\boldsymbol{\theta}) \Biggl) \\
&= \nabla_{\boldsymbol{\theta}} \Biggl( \langle \ \boldsymbol{\theta^T}\boldsymbol{\phi}(\boldsymbol{x})\rangle_{p(\boldsymbol{x})} + A(\boldsymbol{\theta}) + \text{const} \Biggl) \\
&= - \langle \boldsymbol{\phi}(\boldsymbol{x})\rangle_{p(\boldsymbol{x})} + A(\boldsymbol{\theta}) = 0 \\
A(\boldsymbol{\theta}) &= \langle \boldsymbol{\phi}(\boldsymbol{x})\rangle_{p(\boldsymbol{x})}
\end{aligned}
$$
The $\text{const}$ collects all terms not dependent on $\boldsymbol{\theta}$. And, from the result in the previous section we have that

$$
\Big \langle \boldsymbol{\phi}(\boldsymbol{x}) \Big\rangle_{q(\boldsymbol{x})} = \Big \langle \boldsymbol{\phi}(\boldsymbol{x}) \Big\rangle_{p(\boldsymbol{x})}
$$

Working out the matrix of second derivatives will lead to the realization that, at the solution, call it $\boldsymbol{\theta}^*$, we'll be left with a covariance matrix that is positive semi-definite. This completes the proof (read more [here](https://www.herbrich.me/papers/KL.pdf)).

# Conclusion

In the case of a Gaussian distribution, Theorem 1 reduces to matching the mean and covariance via the first and the second moments derived by differentiation of the log partition function. I don't know about you, but "the provided it can be normalized, because the expected statistics can be related to the derivatives of the normalization coefficient" makes a lot of sense now. This result is powerful. It makes me deeply appreciate how profound the exponential family is.