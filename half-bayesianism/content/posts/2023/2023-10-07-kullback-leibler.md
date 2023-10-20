---
title: The unreasonable power of the KL divergence
author: Ewerton de Oliveira
subtitle: A few ideas on the power of KL divergence
description: A few ideas on the power of KL divergence.
date: 2023-10-07
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
draft: false
---
When doing Bayesian Inference, or basically any other type of inference,
chances are you have heard about **KL divergence** -- short for Kullback-Leibler divergence.
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

If you have been exposed to the concept of **surprise** ($I_P(x)=-\log(P(x))$) in Information Theory before, you may realize that equation (2) looks very similar to expected surprise, a.k.a. **entropy**:

\begin{equation}
H_{P(X)}= - \sum_{x \in \mathcal{X}}P(x)\log{\left(P(x)\right)} = \mathbb{E}_P\[I_P(X)\]
\end{equation}

The resemblance is uncanny! This gives us a nice clue to build up intuition. Okay. But... what is entropy then? Don't worry. We certainly don't need to go depth first here and make a full exploration of what entropy means so to later come back to KL. The key point that will allow us to make the connection we are looking for is the fact that entropy is often seen as average information/surprise. This is enough for us here. Just think about what would you get by trying to calculate the difference between $P$ and $Q$ in terms of their average information. Perhaps, you would follow your intuition from physics and decide to do a "delta subtraction":

\begin{equation}
H_{Q(X)} - H_{P(X)} =  \big[ - \sum_{x \in \mathcal{X}}Q(x)\log{\left(Q(x)\right)} \big] - \big[ - \sum_{x \in \mathcal{X}}P(x)\log{\left(P(x)\right)} \big]
\end{equation}

Nice try! KL would be almost that. The catch is that the divergence, $D_{KL}(P||Q)$, is w.r.t $P(X)$. Therefore the correct version is:

\begin{equation}
\begin{aligned}
D_{KL}(P||Q) & = \big[ - \sum_{x \in \mathcal{X}}P(x)\log{\left(Q(x)\right)} \big] - \big[ - \sum_{x \in \mathcal{X}}P(x)\log{\left(P(x)\right)} \big] \newline
& = - \sum_{x \in \mathcal{X}}P(x) \big[ \log{Q(x)} - \log{P(x)} \big] \newline
& = - \sum_{x \in \mathcal{X}}P(x)\log{\left(\frac{Q(x)}{P(x)}\right)}
\end{aligned}
\end{equation}

, which should make you realize the following:

\begin{equation}
D_{KL}(P||Q) = \mathbb{E}_P\[I_Q(X) - I_P(X)\]
\end{equation}

That is my favorite way of thinking about KL: As a difference of **expected surprises** of a "challenging" distribution $Q$ and a reference one, $P$. This last distribution is the first argument of the KL, i.e., $P$ in $D_{KL}(P||Q)$ or $Q$ in $D_{KL}(Q||P)$.

The further apart $Q$ is from $P$ in $D_{KL}(P||Q)$, the worse the former is for the later, and the more surprised one will be when using $Q$ in the reality imposed by $P$.

This is a convenient way of thinking about KL and also explains why $D_{KL}(P||Q)$ is not the same as $D_{KL}(Q||P)$, i.e., why KL is not symmetrical. The metric has wildly different behaviors depending on the relative difference of the distributions. For example, if $P(X) > > Q(X)$ and $Q(X) \approx 0$, then $D_{KL}(P||Q)$ blows up given that $Q$ would be assigning very low probability to frequent events in $P$, hence resulting in higher surprise. That would not happen if $Q(X) > > P(X)$ and $P(X) \approx 0$, given that "surprises" would be in check.

There may be a lot to consider, I know, but let's take a moment to review where we are. By now, we should recognize that the KL divergence between a probability distribution $Q(x)$ and a reference distribution $P(x)$ is
given as follow:

\begin{equation}
\begin{aligned}
D_{KL}(P||Q) & = \big[ - \sum_{x \in \mathcal{X}}P(x)\log{\left(Q(x)\right)} \big] - \big[ - \sum_{x \in \mathcal{X}}P(x)\log{\left(P(x)\right)} \big]  \newline
&= H_{(P(x), Q(x))} − H_{P(x)}
\end{aligned}
\end{equation}

, where $H_{P(x)}$ is the entropy of $P(x)$ and $H_{(P(x), Q(x))}$ is
the **cross entropy** between $P(x)$ and $Q(x)$. One way to think about cross entropy is as the expected surprise of drawing from $P(x)$, when modeling it as $Q(x)$. And yet, the surprise of using $Q$ in the "reality" defined by $P$. And yet, as if "$Q(x)$ would be a map in the territory defined by $P(x)$"... and yet...

That's it! To measure unnecessary surprise from approximating $P(x)$ by $Q(x)$ we use KL!

This really drive everything home! And $(7)$ shall scream the fact that $H_{(P(x),Q(x))} \ge H_{P(x)}$, because a suboptimal model $Q(x)$ will (on average) surprise us more than the reference model $P(x)$. That's where the concept of "divergence" comes from. We're just trying to describe the additional surprise $Q$ will add in w.r.t the reference $P$.

It's then time to see an important property: **KL is always positive!**; There are several ways to prove it. Here, we can do that using two observations. First, if $Q$ doesn't add any average surprise in the reality defined by $P$, then it follows that $H_{(P(x),Q(x))} = H_{P(x)}$. Second, we need to see whether the cross entropy of $P$ and $Q$ can be smaller than the entropy of $P$. It turns out, [Gibb's Inequality](https://en.wikipedia.org/wiki/Gibbs'_inequality), has the answer:

\begin{equation}
-\sum_{x \in \mathcal{X}}P(x)\log{\left(P(x)\right)} \le -\sum_{x \in \mathcal{X}}P(x)\log{\left(Q(x)\right)}
\end{equation}

Then, we have:

\begin{equation}
\begin{aligned}
0 &\le -\sum_{x \in \mathcal{X}}P(x)\log{\left(Q(x)\right)} + \sum_{x \in \mathcal{X}}P(x)\log{\left(P(x)\right)} \newline
&\le \sum_{x \in \mathcal{X}}P(x)\log{\left(\frac{P(x)}{Q(x)}\right)} \newline
& \implies \sum_{x \in \mathcal{X}}P(x)\log{\left(\frac{P(x)}{Q(x)}\right)}  \ge 0
\end{aligned}
\end{equation}

You can also achieve the same result by using the fact that $-D_{KL}(P||Q) \le 0 \implies D_{KL}(P||Q) \ge 0$:

\begin{equation}
\begin{aligned}
-D_{KL}(P||Q) &= - \sum_{x \in \mathcal{X}}P(x)\log{\left(\frac{P(x)}{Q(x)}\right)} \newline
& = \sum_{x \in \mathcal{X}}P(x)\log{\left(\frac{Q(x)}{P(x)}\right)} \newline
& \le \sum_{x \in \mathcal{X}}P(x)\left(\frac{Q(x)}{P(x)}  - 1 \right) \newline
& \le \sum_{x \in \mathcal{X}} Q(x) - \sum_{x \in \mathcal{X}} P(x) \newline
& \le (1 - 1) \newline
& \le 0
\end{aligned}
\end{equation}

, where the first inequality comes fom the fact the $\log{x} \le x − 1$ for all $x > 0$.

Well; There you have it! From this point on you're good to explore [other points of view](https://www.lesswrong.com/posts/no5jDTut5Byjqb4j5/six-and-a-half-intuitions-for-kl-divergence) to KL and build an even stronger intuition, which also includes [other proofs](https://stats.stackexchange.com/questions/335197/why-kl-divergence-is-non-negative). But, before we wrap up the post, we better explore the unreasonable power of KL in inference. Let's explore how it had been used to define an important bound for inference.

Let's say you have a distribution $P(\boldsymbol{Z}|\boldsymbol{X})$ of observed random variables $\boldsymbol{X}$  and latent variables $\boldsymbol{Z}$. In this case the latent variables.
