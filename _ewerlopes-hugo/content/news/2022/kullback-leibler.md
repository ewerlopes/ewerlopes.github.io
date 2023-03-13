+++
date = "2022-06-17T16:06:00-03:00"
title = "The unreasonable power of the KL divergence"
description = "A few ideas on the power of KL divergence."
draft = false
math = true
largeimage = false
+++

<p class=textornament>
    
</p>

<figure style="float: center;" class=ornament-left>
    <img src="/ornaments/roman-ornament.png" title="Speltz, Alexander / Styles of ornament: exhibited in designs, and arranged in historical order, with descriptive text. ([1906]) The Roman ornament. Credits: digicoll.library.wisc.edu"/>
</figure>

<span class="newthought">When doing Bayesian Inference</span>
<span><label for="sn-1" class="margin-toggle sidenote-number">
</span>,
<input type="checkbox" id="sn-1" class="margin-toggle"/>
<span class="sidenote">Or basically any other type of inference. 😆</span>
chances are you may have bumped into a quantity known as <em>KL divergence</em> -- short for Kullback-Leibler divergence.
This quantity has been pervasive in machine learning and artificial intelligence and in this post I would like to explore with you, the reader, some reasons for why this is so, especially for probabilistic inference. Eventually, I am going to enter in the use of the divergence in the framework of Variational Inference and open up the way to talk about how we can do approximate inference with it by means of Expectation Propagation. But first, let's start as usual: with a definition.

What is KL anyway? You might ask. For starters, this is how we compute it:

\begin{equation}
D_{KL}(P||Q) = \sum_{x \in \mathcal{X}}P(x)\log{\left(\frac{P(x)}{Q(x)}\right)}
\end{equation}

, where $P$ and $Q$ are two discrete probability distributions defined on the same probability space $\mathcal{X}$. Equivalently, we often use<span><label for="sn-2" class="margin-toggle sidenote-number">
</span>
<input type="checkbox" id="sn-2" class="margin-toggle"/>
<span class="sidenote">As usual, integrals replace the summation for continuous random variables, where $P$ and $Q$ will be densities.</span>:

\begin{equation}
D_{KL}(P||Q) = - \sum_{x \in \mathcal{X}}P(x)\log{\left(\frac{Q(x)}{P(x)}\right)}
\end{equation}

Well, I know what you may be thinking: "Here comes another handwavy explanation of a complex topic". Nah. Let's try to understand the origins of the expression and see if we can get better intuition. If you have been exposed to the concept of Entropy in Information Theory, you may realize that equation (2) looks like it:

\begin{equation}
H_{P(X)}= - \sum_{x \in \mathcal{X}}P(x)\log{\left(P(x)\right)}
\end{equation}

The resemblance is uncanny! This gives us a nice clue to work with. Okay. But... what is entropy then? Don't worry. We certainly don't need to go depth first here and make a full exploration of what entropy means so to later come back to KL. The key point that will allow us to make the connection we are looking for is the fact entropy is often seen as average information. This is enough for us to build some intuition. Just think about what would you get by trying to calculate the distance between $P$ and $Q$ in terms of their average information. Perhaps, you would follow your intuition from physics and decide to do a "delta subtraction":

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
