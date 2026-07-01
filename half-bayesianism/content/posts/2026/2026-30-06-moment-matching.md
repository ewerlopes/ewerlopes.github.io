---
title: On Moment Matching
date: 2026-06-30
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
draft: false
---

I am not sure about you, but I have reread this passage from Chris Bishop's Pattern Recognition and Machine Learning more times than I care to admit:

> "More generally, it is straightforward to obtain the required expectations for any member of the exponential family, provided it can be normalized, because the expected statistics can be related to the derivatives of the normalization coefficient, as given by (2.226)." (Chapter 10, p. 508)

Every time I see a phrase like "it is straightforward" in a statistics or machine learning text, my first reaction is to brace for impact. Bishop is no exception. The sentence was familiar, yet it never quite stuck in my mind. The ideas of "obtaining the required expectations," "provided it can be normalized," and "derivatives of the normalization coefficient" felt as if they belonged to different worlds. I eventually spent a good deal of time, between the late-night cries of my daughter, trying to make sense of it. The result was a much clearer picture of why the exponential family is so useful in approximate inference.

## Preliminaries

The key idea here is *moment matching*, a technique that is central to approximate inference when the approximating distribution is constrained to lie in the exponential family. Expectation Propagation, which Bishop discusses in the chapter from which that passage comes, relies heavily on repeated local updates that match moments of the approximating distribution.

To make sense of it all, we need to recall the core concepts of the exponential family. In short, a probability distribution belongs to the exponential family if it can be written as

\[
\begin{aligned}
p(\boldsymbol{x}|\boldsymbol{\theta}) &= \frac{1}{Z(\boldsymbol{\theta})}h(\boldsymbol{x})\exp\{\boldsymbol{\theta}^{T}\boldsymbol{\phi}(\boldsymbol{x})\} \\
&= h(\boldsymbol{x})\exp\{\boldsymbol{\theta}^{T}\boldsymbol{\phi}(\boldsymbol{x}) - A(\boldsymbol{\theta})\}
\end{aligned}
\]

where

- \(h(\boldsymbol{x})\) is the base measure (or carrier measure), which captures the distribution's underlying volume;
- \(Z(\boldsymbol{\theta})\) is the partition function, or normalizing constant, which ensures that the density integrates to one;
- \(\boldsymbol{\theta}\) is the vector of natural parameters, with \(\boldsymbol{\theta} \in \mathbb{R}^d\);
- \(\boldsymbol{\phi}(\boldsymbol{x})\) is the vector of sufficient statistics;
- \(A(\boldsymbol{\theta}) = \log Z(\boldsymbol{\theta})\) is the log-partition function.

## Differentiating the log-partition function

The partition function plays a central role in probabilistic modeling. We encounter it whenever we need to make predictions, compare models, or compute expectations. In Bishop's sentence, the "expectations" and "derivatives" are connected through the log-partition function.

Let us start from the fact that a properly normalized exponential-family member integrates to one. Under standard regularity conditions, we can differentiate under the integral sign:

\[
\begin{equation}
\begin{aligned}
\int p(\boldsymbol{x}|\boldsymbol{\theta}) d\boldsymbol{x} &=  \int h(\boldsymbol{x})\exp\{\boldsymbol{\theta}^{T}\boldsymbol{\phi}(\boldsymbol{x}) - A(\boldsymbol{\theta})\}d\boldsymbol{x} = 1 \\
\end{aligned}
\end{equation}
\]

Differentiating both sides with respect to (\boldsymbol{\theta}) gives

\[
\begin{equation}
\begin{aligned}
\nabla \left( \int p(\boldsymbol{x}|\boldsymbol{\theta}) d\boldsymbol{x} \right) &= \nabla \left( \int h(\boldsymbol{x})\exp\{\boldsymbol{\theta}^{T}\boldsymbol{\phi}(\boldsymbol{x}) - A(\boldsymbol{\theta})\}d\boldsymbol{x} \right) = 0
\end{aligned}
\end{equation}
\]

Using the [Leibniz integral rule](https://en.wikipedia.org/wiki/Leibniz_integral_rule), we obtain

\[
\begin{equation}
\begin{aligned}
\nabla \int p(\boldsymbol{x}|\boldsymbol{\theta}) d\boldsymbol{x} &= \nabla \left( \int h(\boldsymbol{x})\exp\{\boldsymbol{\theta}^{T}\boldsymbol{\phi}(\boldsymbol{x}) - A(\boldsymbol{\theta})\}d\boldsymbol{x} \right)\\
&= \int \nabla \left( h(\boldsymbol{x})\exp \left\{\boldsymbol{\theta}^{T}\boldsymbol{\phi}(\boldsymbol{x}) - A(\boldsymbol{\theta})\right\} \right)d\boldsymbol{x} \\
&= \int \underbrace{h(\boldsymbol{x})\exp \left\{\boldsymbol{\theta}^{T}\boldsymbol{\phi}(\boldsymbol{x})- A(\boldsymbol{\theta}) \right\} }_{p(\boldsymbol{x}|\boldsymbol{\theta})} \left(\boldsymbol{\phi}(\boldsymbol{x}) - \nabla A(\boldsymbol{\theta})\right) d\boldsymbol{x} \\
&= \int p(\boldsymbol{x}|\boldsymbol{\theta})\boldsymbol{\phi}(\boldsymbol{x})d\boldsymbol{x} - \int p(\boldsymbol{x}|\boldsymbol{\theta}) \nabla A(\boldsymbol{\theta}) d\boldsymbol{x} \\
&= \mathbb{E}_{p(\boldsymbol{x}|\boldsymbol{\theta})}[\boldsymbol{\phi}(\boldsymbol{x})] - \nabla A(\boldsymbol{\theta}) \int p(\boldsymbol{x}|\boldsymbol{\theta})d\boldsymbol{x} \\
&= \mathbb{E}_{p(\boldsymbol{x}|\boldsymbol{\theta})}[\boldsymbol{\phi}(\boldsymbol{x})] - \nabla A(\boldsymbol{\theta}) \cdot 1 \\
&= 0
\end{aligned}
\end{equation}
\]

Therefore,

\[
\nabla A(\boldsymbol{\theta}) = \mathbb{E}_{p(\boldsymbol{x}|\boldsymbol{\theta})}[\boldsymbol{\phi}(\boldsymbol{x})].
\]

This is the key result. It tells us that the gradient of the log-partition function gives the expectation of the sufficient statistics. In other words, once a distribution is normalized, its moments can often be obtained by differentiation rather than by direct integration.

This is why the normalization coefficient matters so much. It is not just a technical convenience; it encodes the geometry of the distribution and makes its expectations accessible through calculus. A simple Gaussian example makes this concrete: for a Gaussian, the log-partition function is quadratic in the natural parameters, and its first and second derivatives recover the mean and covariance. In that sense, the "normalization coefficient" is really the object that turns probabilistic normalization into a tractable calculus problem.

As an exercise, you can differentiate again to discover that the covariance of \(\boldsymbol{\phi}(\boldsymbol{x})\) can be expressed in terms of the Hessian of \(A(\boldsymbol{\theta})\). Higher-order moments similarly relate to higher-order derivatives.

> "Thus, provided we can normalize a distribution from the exponential family, we can always find its moments by simple differentiation." (Chapter 2, p. 116)

That is the heart of the matter. The normalization constant is the object that allows us to recover the distribution's expectations through differentiation.

## Why "moment matching"?

The KL divergence is central to posterior approximation methods such as variational inference and expectation propagation. This result is especially clean for full exponential-family approximations, whereas mean-field variational inference typically uses a factorized family and a different optimization structure. In the exponential-family case, the connection to moment matching becomes especially clear.

The following theorem captures the key idea.

> **THEOREM 1:** For a target distribution \(p(\boldsymbol{x}|\boldsymbol{\theta})\),and an exponential-family approximation \(q(\boldsymbol{x}|\boldsymbol{\theta}^*)\) with natural sufficient statistics \(\boldsymbol{\phi}(\boldsymbol{x})\), the parameter \(\boldsymbol{\theta}^*\) that minimizes \(KL(p(\boldsymbol{x}|\boldsymbol{\theta})||q(\boldsymbol{x}|\boldsymbol{\theta}^*))\) satisfies
> \( \mathbb{E}_{q(\boldsymbol{x}|\boldsymbol{\theta}^*)}[\boldsymbol{\phi}(\boldsymbol{x})] = \mathbb{E}_{p(\boldsymbol{x}|\boldsymbol{\theta})}[\boldsymbol{\phi}(\boldsymbol{x})]\).

In other words, the optimal distribution in the exponential family is the one whose expected sufficient statistics match those of the target distribution. This is why the term "moment matching" is used. The "moments" here are the expectations of the sufficient statistics, and the "matching" refers to the equality above.

To see why this is true, let us write the KL divergence as a function of the parameters \(\boldsymbol{\theta}\). I will also simplify the distribution notation by removing the reference to its latent parameters.

\[
\begin{equation}
\begin{aligned}
f(\boldsymbol{\theta}) &= KL(p(\boldsymbol{x})||q(\boldsymbol{x}))
= \mathbb{E}_{p(\boldsymbol{x})}\left[\ln \left( \frac{p(\boldsymbol{x})}{q(\boldsymbol{x})}\right)\right]  \\
&= \int p(\boldsymbol{x}) \ln p(\boldsymbol{x}) d\boldsymbol{x} - \int p(\boldsymbol{x}) \ln q(\boldsymbol{x}) d\boldsymbol{x} \\
&= \int p(\boldsymbol{x}) \ln p(\boldsymbol{x}) d\boldsymbol{x} - \int p(\boldsymbol{x}) \ln \left( h(\boldsymbol{x}) \exp \left\{ \boldsymbol{\theta}^T\boldsymbol{\phi}(\boldsymbol{x}) - A(\boldsymbol{\theta}) \right\} \right) d\boldsymbol{x} \\
&= \int p(\boldsymbol{x}) \ln p(\boldsymbol{x}) d\boldsymbol{x} - \int p(\boldsymbol{x}) \ln h(\boldsymbol{x}) d\boldsymbol{x} - \int p(\boldsymbol{x}) \boldsymbol{\theta}^T\boldsymbol{\phi}(\boldsymbol{x}) d\boldsymbol{x} + \int p(\boldsymbol{x}) A(\boldsymbol{\theta}) d\boldsymbol{x}
\end{aligned}
\end{equation}
\]
We then need to compute the gradient w.r.t. \(\boldsymbol{\theta}\) and set it to zero

\[
\begin{equation}
\begin{aligned}
\nabla_{\boldsymbol{\theta}} f(\boldsymbol{\theta}) &= \nabla_{\boldsymbol{\theta}} \left( \mathbb{E}_{p(\boldsymbol{x})}[\ln(p(\boldsymbol{x}))] - \mathbb{E}_{p(\boldsymbol{x})}[\ln(h(\boldsymbol{x}))] - \mathbb{E}_{p(\boldsymbol{x})}[\boldsymbol{\theta}^T\boldsymbol{\phi}(\boldsymbol{x})] + A(\boldsymbol{\theta}) \right) \\
&= \nabla_{\boldsymbol{\theta}} \left( \mathbb{E}_{p(\boldsymbol{x})}[\boldsymbol{\theta}^T\boldsymbol{\phi}(\boldsymbol{x})] + A(\boldsymbol{\theta}) + \text{const} \right) \\
&= - \mathbb{E}_{p(\boldsymbol{x})}[\boldsymbol{\phi}(\boldsymbol{x})] + \nabla A(\boldsymbol{\theta}) = 0 \\
\nabla A(\boldsymbol{\theta}) &= \mathbb{E}_{p(\boldsymbol{x})}[\boldsymbol{\phi}(\boldsymbol{x})]
\end{aligned}
\end{equation}
\]
The \(\text{const}\) collects all terms not dependent on \(\boldsymbol{\theta}\). And, from the result in the previous section we have that

\[
\begin{equation}
\mathbb{E}_{q(\boldsymbol{x})}[\boldsymbol{\phi}(\boldsymbol{x})] = \mathbb{E}_{p(\boldsymbol{x})}[\boldsymbol{\phi}(\boldsymbol{x})]
\end{equation}
\]

The second derivative of the objective is positive semidefinite, so this stationary point is a minimum. That completes the argument, assuming the relevant expectations are finite and the optimum exists. This completes the proof (read more [here](https://www.herbrich.me/papers/KL.pdf)).

## Conclusion

In the Gaussian case, Theorem 1 reduces to matching the mean and covariance, because the first and second derivatives of the log-partition function yield the first and second moments. That is why the statement "the expected statistics can be related to the derivatives of the normalization coefficient" now makes much more sense. The result is powerful, and it highlights one of the great strengths of the exponential family: its members are not merely convenient parametric forms, but families for which expectations can be recovered through differentiation.

In practice, this perspective is useful because it turns the problem of approximate inference into a problem of matching a few summary statistics. That is exactly why exponential-family approximations are so common in Bayesian modeling, variational methods, and expectation propagation.

Enough said. I need to go see why my daughter is crying again.

> postscript: This post was initially drafted in July 2025, but I never got around to finishing it. I am now revisiting it in June 2026, to get some break from the generative AI discourse dominance. Please, send me an email if you find any mistakes or have suggestions for improvement. BTW, this post was lightly prof-read by AI. All derivations were checked by hand, but I cannot guarantee that there are no typos or mistakes. =)
