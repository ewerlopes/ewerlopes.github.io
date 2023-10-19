---
title: The unreasonable power of the KL divergence
author: Ewerton de Oliveira
subtitle: A few ideas on the power of KL divergence
description: A few ideas on the power of KL divergence.
date: 2022-10-07
math: true
categories:
  - bayesian
  - inference
tags:
  - optimization
  - cost-functions
  - divergence-measures
  - generative-ai
  - llm
draft: true
---
When doing Bayesian Inference, or basically any other type of inference,
chances are you have heard about *KL divergence* -- short for Kullback-Leibler divergence.
This quantity has been pervasive in machine learning and artificial intelligence and in this post I would like to explore with you, the reader, some reasons for why this is so, especially for probabilistic inference. Eventually, we'll find ourselves dealing with Variational Inference and with a little more patience, with Expectation Propagation. But first, let's start as usual: with a definition.

What is KL anyway? You might ask. For starters, this is how we compute it:

\begin{equation}
D_{KL}(P||Q) = \sum_{x \in \mathcal{X}}P(x)\log{\left(\frac{P(x)}{Q(x)}\right)}
\end{equation}

, where $P$ and $Q$ are two discrete probability mass functions defined on the same probability space $\mathcal{X}$. Equivalently, we often use:

\begin{equation}
D_{KL}(P||Q) = - \sum_{x \in \mathcal{X}}P(x)\log{\left(\frac{Q(x)}{P(x)}\right)}
\end{equation}

and we would naturally be using integrals to replace the summation for continuous random variables where $P$ and $Q$ would be densities. Well, I know what you may be thinking: "Here comes another handwavy explanation of a complex topic". Nah. Let's try to understand the origins of the expression and see if we can get better intuition.

If you have been exposed to the concept of **surprise**, and ultimately **entropy**, in Information Theory before, you may realize that equation (2) looks kind of familiar.

\begin{equation}
H_{P(X)}= - \sum_{x \in \mathcal{X}}P(x)\log{\left(P(x)\right)}
\end{equation}

The resemblance is uncanny! This already gives us a nice clue to work with. Okay. But... what is entropy then? Don't worry. We certainly don't need to go depth first here and make a full exploration of what entropy means so to later come back to KL. The key point that will allow us to make the connection we are looking for is the fact that entropy is often seen as average information. This is enough for us to build some intuition. Just think about what would you get by trying to calculate the distance between $P$ and $Q$ in terms of their average information. Perhaps, you would follow your intuition from physics and decide to do a "delta subtraction":

\begin{equation}
H_{Q(X)} - H_{P(X)} =  \big[ - \sum_{x \in \mathcal{X}}Q(x)\log{\left(Q(x)\right)} \big] - \big[ - \sum_{x \in \mathcal{X}}P(x)\log{\left(P(x)\right)} \big]
\end{equation}

Nice try! KL would be almost that. The catch is that the divergence, $D_{KL}(P||Q)$, is w.r.t $P(X)$. Therefore the correct version is similar:

\begin{equation}
\begin{aligned}
D_{KL}(P||Q) & =  \big[ - \sum_{x \in \mathcal{X}}P(x)\log{\left(Q(x)\right)} \big] - \big[ - \sum_{x \in \mathcal{X}}P(x)\log{\left(P(x)\right)} \big] \newline
& = - \sum_{x \in \mathcal{X}}P(x) \big[ \log{Q(x)} - \log{P(x)} \big] \newline
& = - \sum_{x \in \mathcal{X}}P(x)\log{\left(\frac{Q(x)}{P(x)}\right)}
\end{aligned}
\end{equation}

That is my favorite way of thinking about KL: As a divergence (a.k.a. distance) of average information w.r.t. the first argument of the KL, i.e., $P$ in $D_{KL}(P||Q)$ or $Q$ in $D_{KL}(Q||P)$.

Ok. Where are we so far? You would agree that computing something is very different from knowing the reasons for it. That is to say that knowing how to compute something does not necessarily mean we understand from where the quantity in question came from. That is the challenge now. What is the origin of KL? Can we derive it somehow?


