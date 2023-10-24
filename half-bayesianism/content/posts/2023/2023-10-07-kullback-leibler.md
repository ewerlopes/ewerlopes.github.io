---
title: The reasonable power of the KL divergence
author: Ewerton de Oliveira
subtitle: From intuition to deterministic inference
description: A few ideas on the power of KL divergence.
date: 2023-10-24
math: true
categories:
  - bayesian
  - inference
  - divergence-measures
tags:
  - optimization
  - cost-functions
  - generative-ai
draft: false
---
When doing Bayesian Inference, or any other type of inference,
chances are you have heard about **KL divergence** -- short for [Kullback-Leibler divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence).
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

The resemblance is uncanny! This gives us a nice clue to build up intuition. Okay. But... what is entropy then? Don't worry. We certainly don't need to go in-depth here and make a full exploration of what entropy means to later come back to KL. The key point that will allow us to make the connection we are looking for is the fact that entropy is often seen as average information/surprise. This is enough for us here. Just think about what would you get by trying to calculate the difference between $P$ and $Q$ in terms of their average information. Perhaps, you would follow your intuition from physics and decide to do a "delta subtraction":

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

That is my favourite way of thinking about KL: As a difference between **expected surprises** of a "challenging" distribution $Q$ and a reference one, $P$. This last distribution is the first argument of the KL, i.e., $P$ in $D_{KL}(P||Q)$ or $Q$ in $D_{KL}(Q||P)$.

The further apart $Q$ is from $P$ in $D_{KL}(P||Q)$, the worse the former is for the latter, and the more surprised one will be when using $Q$ in the reality imposed by $P$.

This is a convenient way of thinking about KL and also explains why $D_{KL}(P||Q)$ is not the same as $D_{KL}(Q||P)$, i.e., why KL is not symmetrical. The metric has wildly different behaviours depending on the relative difference of the distributions. For example, if $P(X) > > Q(X)$ and $Q(X) \approx 0$, then $D_{KL}(P||Q)$ blows up given that $Q$ would be assigned a very low probability to frequent events in $P$, hence resulting in higher surprise. That would not happen if $Q(X) > > P(X)$ and $P(X) \approx 0$, given that "surprises" would be in check.

There may be a lot to consider, I know, but let's take a moment to review where we are. By now, we should recognize that the KL divergence between a probability distribution $Q(x)$ and a reference distribution $P(x)$ is
given as follows:

\begin{equation}
\begin{aligned}
D_{KL}(P||Q) & = \big[ - \sum_{x \in \mathcal{X}}P(x)\log{\left(Q(x)\right)} \big] - \big[ - \sum_{x \in \mathcal{X}}P(x)\log{\left(P(x)\right)} \big]  \newline
&= H_{(P(x), Q(x))} − H_{P(x)}
\end{aligned}
\end{equation}

, where $H_{P(x)}$ is the entropy of $P(x)$ and $H_{(P(x), Q(x))}$ is
the **cross entropy** between $P(x)$ and $Q(x)$. One way to think about cross-entropy is as the expected surprise of drawing from $P(x)$ when modelling it as $Q(x)$. And yet, the surprise of using $Q$ in the "reality" defined by $P$. And yet, as if "$Q(x)$ would be a map in the territory defined by $P(x)$"... and yet...

That's it! To measure unnecessary surprise by approximating $P(x)$ by $Q(x)$ we use KL! This is one of the reasons why the divergence is so ubiquitous.

This drives everything home! And $(7)$ shall scream the fact that $H_{(P(x), Q(x))} \ge H_{P(x)}$ because a suboptimal model $Q(x)$ will (on average) surprise us more than the reference model $P(x)$. That's where the concept of "divergence" comes from. We're just trying to describe the additional surprise $Q$ will add in w.r.t the reference $P$.

It's then time to see an important property: **KL is always positive!**; There are several ways to prove it. Here, we can do that using two observations. First, if $Q$ doesn't add any average surprise in the reality defined by $P$, then it follows that $H_{(P(x), Q(x))} = H_{P(x)}$. Second, we need to see whether the cross entropy of $P$ and $Q$ can be smaller than the entropy of $P$. It turns out, [Gibb's Inequality](https://en.wikipedia.org/wiki/Gibbs'_inequality), has the answer:

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

From this point on you're good to explore [other points of view](https://www.lesswrong.com/posts/no5jDTut5Byjqb4j5/six-and-a-half-intuitions-for-kl-divergence) and [other proofs](https://stats.stackexchange.com/questions/335197/why-kl-divergence-is-non-negative) to KL and build an even stronger intuition for its pervasiveness. Before that though, we better explore it a bit more, via some inference mechanism. Let's see how it has been used to define an important bound: the ELBO, short for "Evidence Lower Bound".

Let's say you have a posterior, $P(\boldsymbol{Z}|\boldsymbol{X})$, of observed random variables $\boldsymbol{X}$ and latent variables $\boldsymbol{Z}$ -- they are taken as sets of variables in case you did not notice the boldface 😊. You may recognize this as the central task in probabilistic inference: model the behaviour of latent variables of interest via expectations computed w.r.t. the distribution of the observed variables.

Sometimes, however, the computation of the posterior is intractable due to several factors. Among the most popular reasons, are the lack of analytical form for highly complex required integrals in the posterior and high dimensionality leading to costly numerical integrations.

In another post, we will start digging into "Variational Inference" as a deterministic form of approximated inference in probabilistic models. For now, it suffices to consider the posterior $P(\boldsymbol{Z}|\boldsymbol{X})$ as intractable and opt for some approximation with a simpler distribution. Guess what?! KL can help! We'll opt for using a distribution $Q(\boldsymbol{Z})$ to estimate our posterior. For that, we need a measure of the quality of the estimation, and we are gonna see that KL is a good candidate for it. Here is how we begin:

\begin{equation}
D_{KL}(Q(\boldsymbol{Z})||P(\boldsymbol{Z}|\boldsymbol{X})) = - \sum_{x \in \mathcal{Z}}Q(\boldsymbol{Z})\log{\left(\frac{P(\boldsymbol{Z}|\boldsymbol{X})}{Q(\boldsymbol{Z})}\right)}
\end{equation}

Therefore, using the conditional probability rule:

\begin{equation}
\begin{aligned}
D_{KL}(Q(\boldsymbol{Z})||P(\boldsymbol{Z}|\boldsymbol{X})) &= - \sum_{x \in \mathcal{Z}}Q(\boldsymbol{Z})\log{\left(\frac{P(\boldsymbol{Z},\boldsymbol{X})}{P(\boldsymbol{X})}\cdot\frac{1}{Q(\boldsymbol{Z})}\right)}
\newline
&= - \sum_{x \in \mathcal{Z}}Q(\boldsymbol{Z})\log{\left(\frac{P(\boldsymbol{Z},\boldsymbol{X})}{Q(\boldsymbol{Z})}\cdot\frac{1}{P(\boldsymbol{X})}\right)}
\newline
&= - \sum_{x \in \mathcal{Z}}Q(\boldsymbol{Z})\left(\log{\left(\frac{P(\boldsymbol{Z},\boldsymbol{X})}{Q(\boldsymbol{Z})}\right)} + \log{\left(\frac{1}{P(\boldsymbol{X})}\right)} \right)
\newline
&= - \sum_{x \in \mathcal{Z}}Q(\boldsymbol{Z})\left(\log{\left(\frac{P(\boldsymbol{Z},\boldsymbol{X})}{Q(\boldsymbol{Z})}\right)} - \log{\left(P(\boldsymbol{X})\right)} \right)
\newline
&= - \sum_{x \in \mathcal{Z}}Q(\boldsymbol{Z})\log{\left(\frac{P(\boldsymbol{Z},\boldsymbol{X})}{Q(\boldsymbol{Z})}\right)} + \sum_{x \in \mathcal{Z}}Q(\boldsymbol{Z})\log{\left(P(\boldsymbol{X})\right)}
\newline
&= - \sum_{x \in \mathcal{Z}}Q(\boldsymbol{Z})\log{\left(\frac{P(\boldsymbol{Z},\boldsymbol{X})}{Q(\boldsymbol{Z})}\right)} + \log{\left(P(\boldsymbol{X})\right)}
\end{aligned}
\end{equation}

, where we have used the fact that the summation of the entire support of $Q(Z)$ equals $1$ in the last equality.

We now isolate $P(\boldsymbol{X})$:

\begin{equation}
\begin{aligned}
D_{KL}(Q(\boldsymbol{Z})||P(\boldsymbol{Z}|\boldsymbol{X})) + \sum_{x \in \mathcal{Z}}Q(\boldsymbol{Z})\log{\left(\frac{P(\boldsymbol{Z},\boldsymbol{X})}{Q(\boldsymbol{Z})}\right)} &= \log{\left(P(\boldsymbol{X})\right)}
\newline
D_{KL}(Q(\boldsymbol{Z})||P(\boldsymbol{Z}|\boldsymbol{X})) - D_{KL}(Q(\boldsymbol{Z})||P(\boldsymbol{Z},\boldsymbol{X})) &= \log{\left(P(\boldsymbol{X})\right)}
\end{aligned}
\end{equation}

We notice interesting points in $(13)$:

* The first term is always positive (By $(9)$ and $(10)$).
* The second term, called the **Evidence Lower Bound**, is always negative.
* The RHS is always less than or equal to zero since $P(\boldsymbol{X})$ is a distribution and $log(x) \le 0$ for $x \in [0, 1]$. From $P(\boldsymbol{Z}|\boldsymbol{X})$ it's given and therefore fixed.

For convenience, we will rename the terms in $(13)$ as:

\begin{equation}
D_{KL} + \mathcal{L} = \log{\left(P(\boldsymbol{X})\right)}
\end{equation}

Now, we can observe the behaviour of these quantities by playing with some values exercising what we observed so far and considering the $\log{\left(P(\boldsymbol{X})\right)}$ fixed at $-4$:

|  $D_{KL}$ | $\mathcal{L}$  | $\log{\left(P(\boldsymbol{X})\right)}$  |
|:---------------:|:---------------:|:---------------:|
|  4 | -8  | -4  |
| 3  | -7  | -4  |
| 2  | -6  | -4  |

Therefore, because $P(\boldsymbol{X})$ is fixed in the context of the posterior, $\mathcal{L}$ "controls the KL divergence", acting like a lower bound. This is done via inverse proportion: if one goes up the other must go down. This has an immediate advantage that by making the lower bound $\mathcal{L}$ less negative, i.e. larger, we reduce the KL divergence, which consequently makes the approximation of $P(\boldsymbol{Z}|\boldsymbol{X})$ by $P(\boldsymbol{Z})$ better.

This is a key point. *When approximating the posterior, instead of minimizing the KL divergence with the problematic posterior we can resort to maximizing the lower bound $\mathcal{L}$*. This may not sound like much, but often it's much, much easier to deal with the lower bound than with the KL directly. This is because most of the time we have the joint probability underlying the posterior and in that, we don't need to deal with intractable marginalizations. From the above, we shall be able to understand the basic role of KL in $(12)$. We are measuring the surprise in the approximation, and in doing so we prefer having it low, which would mean a better approximation. Since approximations are pervasive in the computing world, it's reasonable to see that property being carried over to KL.

How you maximize the lower bound is what will instantiate the theory and make you achieve wonderful things. A full adventure into the realm of Bayesian inference via deterministic approximation methods like "Variational Inference" is certainly a huge topic and one I hope to cover in more detail in a separate post. For now, I wanted to give you a glimpse of how KL appears in inference so that you can further develop your intuition about the divergence.

KL is pervasive across the information theory literature (yes, that includes ML, AI, etc.). At the time of this post, generative AI is being discussed everywhere, and all CEOs, engineers and researchers, across a large sector of the working force, are talking about it during their happy hours on a Friday and Saturday night. Even there, you will notice KL's presence. For example, during [Reinforcement Learning with Human Feedback](https://arxiv.org/pdf/2203.02155.pdf) (RLHF), a technique used to improve (fine-tune) Large Language Models in the direction of human feedback, one uses KL to anchor model response adaptation to the previously trained model, avoiding unstable learning grounded in adaptations that are too aggressive. Take a look at the paper and see if you can spot the role of KL there.

Keep learning.