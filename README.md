Premise of GANs
===

A GAN takes a random sample from a latent or prior distribution as input and maps it to the data space. The task of training is to learn a deterministic function that can efficiently capture the dependencies and patterns in the data so that the mapped point resembles a sample generated from the data distribution. 

Example:
---

I have generated 300 samples from `Isotropic Bivariate Gaussian distribution`. 

![bi var guass](Images/bi_var_guassian.png)

When passed through a function <a href="https://www.codecogs.com/eqnedit.php?latex=f(x)&space;=&space;x/10&space;&plus;&space;x/||x||" target="_blank"><img src="https://latex.codecogs.com/svg.latex?f(x)&space;=&space;x/10&space;&plus;&space;x/||x||" title="f(x) = x/10 + x/||x||" /></a>, the points form a `ring`, which demonstrates that there could be a high capacity function that may be able to model data distribution of high dimensional data like images. Neural networks are out best bet as they are universal functional approximators. Hence, deep neural networks are used while modeling data distribution of images. Unlike MLE or KDE, this is an implicit density estimation

![ring](Images/Ring_formation.png)

Probability Review
---

![freq table](Images/frequency_table.png)
*frequency table and joint distribution of two discrete random variables*

<h3>Conditional Distribution</h3> 

In the above table, fix the value of a random variable, say `x = x_1`; the distribution of `y` when `x = x_1` is called conditional distribution, <a href="https://www.codecogs.com/eqnedit.php?latex=P(y&space;|&space;X=x_{1})" target="_blank"><img src="https://latex.codecogs.com/svg.latex?P(y&space;|&space;X=x_{1})" title="P(y | X=x_{1})" /></a>. Conditional expectation is expectation of the conditional distribution. 

In the above table, the conditional probability of `y_1` given `X=x_1` is `2/17`

<h3>Marginal Distribution</h3>

Integrate or summate over a variable, to get the marginal distribution of another variable. 
<a href="https://www.codecogs.com/eqnedit.php?latex=P(X&space;=&space;x_{i})&space;=&space;\sum_{y=1}^{n}&space;P(X&space;=&space;x_{i},&space;Y&space;=&space;y_{i})" target="_blank"><img src="https://latex.codecogs.com/svg.latex?P(X&space;=&space;x_{i})&space;=&space;\sum_{y=1}^{n}&space;P(X&space;=&space;x_{i},&space;Y&space;=&space;y_{i})" title="P(X = x_{i}) = \sum_{y=1}^{n} P(X = x_{i}, Y = y_{i})" /></a>

In the above table, the marginal probability of `x_1` according to above formula is `2/50 + 10/50 + 5/50 = 17/50`. 

<h3>Joint Distribution</h3>

A join distribution a.k.a data distribution captures the joint probabilities between random variables. In the above table, the join probability of `P(X = x_2 & Y = y_3)` is `2/50`. This is what a GAN tries to model from the sample data. 

Consider images of size `28 x 28`. Each pixel is a random variable that can take any value from 0 to 255. Hence, we have 784 random variables in total. GAN tries to model the dependencies between the pixels. 

<h3>Bayes Theorm</h3>

<a href="https://www.codecogs.com/eqnedit.php?latex=P(&space;A&space;|&space;B)&space;=&space;\frac{P(A&space;\bigcap&space;B)}{P(B))}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?P(&space;A&space;|&space;B)&space;=&space;\frac{P(A&space;\bigcap&space;B)}{P(B))}" title="P( A | B) = \frac{P(A \bigcap B)}{P(B))}" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=Conditional&space;Probability&space;=&space;\frac{Joint&space;Probability}{Marginal&space;Probability}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?Conditional&space;Probability&space;=&space;\frac{Joint&space;Probability}{Marginal&space;Probability}" title="Conditional Probability = \frac{Joint Probability}{Marginal Probability}" /></a>

From the above table: `P(Y = y_1 | X = x_1) = P(Y = y_1 & X = x_1) / P(X = x_1)` = `(2/50)/(17/50)` = `2/17`

<h3>Entropy</h3>

Entropy measures the degree of uncertainty of an outcome of a trial according to a `p(x)`. 

<a href="https://www.codecogs.com/eqnedit.php?latex=H(p)&space;=&space;-\sum_{k=1}^{K}p_{i}\log&space;p_{i}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?H(p)&space;=&space;-\sum_{k=1}^{K}p_{k}\log&space;p_{k}" title="H(p) = -\sum_{k=1}^{K}p_{i}\log p_{i}" /></a>

![entropy](Images/entropy.png)

 
The entropy of an unbiased coin is higher than a biased coin. The difference in entropy increases with the degree of polarization of probabilities of the biased coin.

<h3>Cross Entropy</h3>

Cross entropy measures the degree of uncertainty of a trial according to `p(y)` but in truth according to `p(x)`.

<a href="https://www.codecogs.com/eqnedit.php?latex=H(p(x),&space;p(y))&space;=&space;-\sum_{k=1}^{K}p(x_{k})\log&space;p(y_{k})" target="_blank"><img src="https://latex.codecogs.com/svg.latex?H(p(x),&space;p(y))&space;=&space;-\sum_{k=1}^{K}p(x_{k})\log&space;p(y_{k})" title="H(p(x), p(y)) = -\sum_{k=1}^{K}p(x_{k})\log p(y_{k})" /></a>

![entropy](Images/cross_entropy.png)

Cross entropy is higher when a trial is conducted according to unbiased coin probability distribution but you think it is being conducted according to the biased coin probability distribution. 

<h3>KL Divergence</h3>

KL Divergence is the difference between cross-entropy and entropy of the true distribution. KL Divergence is equal to zero when two distributions are equal. Hence, when you want to approximate or model a probability distribution with other, minimizing the KL Divergence between them will make them similar.

<a href="https://www.codecogs.com/eqnedit.php?latex=D_{KL}(p&space;||&space;q)&space;=&space;H(p,&space;q)&space;-&space;H(p)&space;=&space;-&space;\sum_{k=1}^{K}&space;p_{k}\log&space;\frac{p_{k}}{q_{k}}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?D_{KL}(p&space;||&space;q)&space;=&space;H(p,&space;q)&space;-&space;H(p)&space;=&space;-&space;\sum_{k=1}^{K}&space;p_{k}\log&space;\frac{p_{k}}{q_{k}}" title="D_{KL}(p || q) = H(p, q) - H(p) = - \sum_{k=1}^{K} p_{k}\log \frac{p_{k}}{q_{k}}" /></a>

`D(fair||biased) = H(fair, biased) – H(fair) = 2.19-1 = 1.19`

<h3>JS Divergence</h3>

Due to division by a probability of an outcome in KL Divergence equation, it may become intractable in some cases. For example, if `q_k` is zero, KL Divergence becomes infinite. Moreover, KL Divergence is not symmetric i.e., `D(p||q)` is not equal to `D(q||p)`, which makes it unusable as a distance metric. To suppress these effects, JS divergence uses avg probability of an outcome.

<a href="https://www.codecogs.com/eqnedit.php?latex=D_{JS}(p&space;||&space;q)&space;=&space;\frac{1}{2}\&space;D_{KL}(p&space;||&space;\frac{p&space;&plus;&space;q}{2})&space;&plus;&space;\frac{1}{2}\&space;D_{KL}(q&space;||&space;\frac{p&space;&plus;&space;q}{2})" target="_blank"><img src="https://latex.codecogs.com/svg.latex?D_{JS}(p&space;||&space;q)&space;=&space;\frac{1}{2}\&space;D_{KL}(p&space;||&space;\frac{p&space;&plus;&space;q}{2})&space;&plus;&space;\frac{1}{2}\&space;D_{KL}(q&space;||&space;\frac{p&space;&plus;&space;q}{2})" title="D_{JS}(p || q) = \frac{1}{2}\ D_{KL}(p || \frac{p + q}{2}) + \frac{1}{2}\ D_{KL}(q || \frac{p + q}{2})" /></a>

GANs
----

As aforementioned, GANs take a random sample from the latent space as input and maps it to data space. In DCGANs, the mapping function is a deep neural network, which is differentiable and parameterized by network weights. The mapping function is called `Generator(G)`. A `Discriminator(D)` is also a deep neural network that takes a sample in the data space and maps it to the action space i.e., the probability of that sample being generated from the data distribution. 

<a href="https://www.codecogs.com/eqnedit.php?latex=P_{z}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?P_{z}" title="P_{z}" /></a> :  Prior / latent distribution. Typically, this space is much smaller than the data space.

<a href="https://www.codecogs.com/eqnedit.php?latex=P_{g}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?P_{g}" title="P_{g}" /></a>: Data distribution of data generated by the generator

<a href="https://www.codecogs.com/eqnedit.php?latex=P_{r}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?P_{r}" title="P_{r}" /></a>: Real data distribution

<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\small&space;\widehat{y_{i}}&space;=&space;D(x_{i})" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\small&space;\widehat{y_{i}}&space;=&space;D(x_{i})" title="\small \widehat{y_{i}} = D(x_{i})" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\small&space;g_{i}&space;=&space;G(z_{i})" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\small&space;g_{i}&space;=&space;G(z_{i})" title="\small g_{i} = G(z_{i})" /></a>

D__m = (d_1, d_2, d_3, ... d_m) be the data generated according to P_r

G__n = (g_m+1, g_m+2, .. g_n) be the data generated according to P_g


Train `D` to minimize the empirical loss. I am including min functions, as most deep learning frameworks only implement, minimization of a function.

<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;L_{d}&space;=&space;\max_{\theta&space;_{d}}\sum_{i=1}^{n}\1(x_{i}&space;\epsilon&space;D_{m})log&space;\widehat{y}_{i}&space;&plus;&space;1(x_{i}\epsilon&space;G_{n})&space;\log&space;(1-&space;\widehat{y}_{i}))&space;=&space;\min_{\theta&space;_{d}}-&space;\sum_{i=1}^{n}\1(x_{i}&space;\epsilon&space;D_{m})log&space;\widehat{y}_{i}&space;&plus;&space;1(x_{i}\epsilon&space;G_{n})&space;\log&space;(1-&space;\widehat{y}_{i}))&space;\rightarrow&space;1" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;L_{d}&space;=&space;\max_{\theta&space;_{d}}\sum_{i=1}^{n}\1(x_{i}&space;\epsilon&space;D_{m})log&space;\widehat{y}_{i}&space;&plus;&space;1(x_{i}\epsilon&space;G_{n})&space;\log&space;(1-&space;\widehat{y}_{i}))&space;=&space;\min_{\theta&space;_{d}}-&space;\sum_{i=1}^{n}\1(x_{i}&space;\epsilon&space;D_{m})log&space;\widehat{y}_{i}&space;&plus;&space;1(x_{i}\epsilon&space;G_{n})&space;\log&space;(1-&space;\widehat{y}_{i}))&space;\rightarrow&space;1" title="L_{d} = \max_{\theta _{d}}\sum_{i=1}^{n}\1(x_{i} \epsilon D_{m})log \widehat{y}_{i} + 1(x_{i}\epsilon G_{n}) \log (1- \widehat{y}_{i})) = \min_{\theta _{d}}- \sum_{i=1}^{n}\1(x_{i} \epsilon D_{m})log \widehat{y}_{i} + 1(x_{i}\epsilon G_{n}) \log (1- \widehat{y}_{i})) \rightarrow 1" /></a>

Fix the `D` network, and train `G` to maximize the loss of `D` over G_n. 

<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\fn_cm&space;\small&space;L_{g}&space;=&space;\max_{\theta&space;_{g}}&space;-&space;\sum_{i=1}^{n}\&space;1(x_{i}\epsilon&space;G_{n})&space;\log&space;(1-&space;\widehat{y}_{i}))&space;=&space;\min_{\theta&space;_{g}}&space;\sum_{i=1}^{n}\&space;1(x_{i}\epsilon&space;G_{n})&space;\log&space;(1-&space;\widehat{y}_{i}))" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\fn_cm&space;\small&space;L_{g}&space;=&space;\max_{\theta&space;_{g}}&space;-&space;\sum_{i=1}^{n}\&space;1(x_{i}\epsilon&space;G_{n})&space;\log&space;(1-&space;\widehat{y}_{i}))&space;=&space;\min_{\theta&space;_{g}}&space;\sum_{i=1}^{n}\&space;1(x_{i}\epsilon&space;G_{n})&space;\log&space;(1-&space;\widehat{y}_{i}))" title="\small L_{g} = \max_{\theta _{g}} - \sum_{i=1}^{n}\ 1(x_{i}\epsilon G_{n}) \log (1- \widehat{y}_{i})) = \min_{\theta _{g}} \sum_{i=1}^{n}\ 1(x_{i}\epsilon G_{n}) \log (1- \widehat{y}_{i}))" /></a>

As stated in the original paper, in the early training period, the above loss doesn't offer enough gradient to update the parameters of `G` network, as initially `P_g` is distant from `P_d`, which makes it easy for `D` to classify generated images. Hence, we try to maximize it by switching labels.

<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\fn_cm&space;\small&space;L_{g}&space;=\min_{\theta&space;_{g}}&space;\sum_{i=1}^{n}\&space;1(x_{i}\epsilon&space;G_{n})&space;\log&space;(1-&space;\widehat{y}_{i}))&space;=&space;\max_{\theta&space;_{g}}&space;\sum_{i=1}^{n}\&space;1(x_{i}\epsilon&space;G_{n})&space;\log&space;(\widehat{y}_{i}))&space;=&space;\min_{\theta&space;_{g}}&space;-&space;\sum_{i=1}^{n}\&space;1(x_{i}\epsilon&space;G_{n})&space;\log&space;(\widehat{y}_{i})" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\fn_cm&space;\small&space;L_{g}&space;=\min_{\theta&space;_{g}}&space;\sum_{i=1}^{n}\&space;1(x_{i}\epsilon&space;G_{n})&space;\log&space;(1-&space;\widehat{y}_{i}))&space;=&space;\max_{\theta&space;_{g}}&space;\sum_{i=1}^{n}\&space;1(x_{i}\epsilon&space;G_{n})&space;\log&space;(\widehat{y}_{i}))&space;=&space;\min_{\theta&space;_{g}}&space;-&space;\sum_{i=1}^{n}\&space;1(x_{i}\epsilon&space;G_{n})&space;\log&space;(\widehat{y}_{i})" title="\small L_{g} =\min_{\theta _{g}} \sum_{i=1}^{n}\ 1(x_{i}\epsilon G_{n}) \log (1- \widehat{y}_{i})) = \max_{\theta _{g}} \sum_{i=1}^{n}\ 1(x_{i}\epsilon G_{n}) \log (\widehat{y}_{i})) = \min_{\theta _{g}} - \sum_{i=1}^{n}\ 1(x_{i}\epsilon G_{n}) \log (\widehat{y}_{i})" /></a>



<h3> Optimization And Theoritical Results</h3>

<h4> Optimal Discriminator for fixed `G` </h4>

Equation `1` is an empiracal loss function. Its risk function or loss on the whole population i.e., for every possible image can be written as: 

<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\small&space;L_{d}^{r}=&space;E_{x\sim&space;p_{r}}\log&space;\widehat{y}\:&space;&plus;&space;E_{x\sim&space;p_{g}}\log&space;1-&space;\widehat{y}&space;=&space;\int_{x}^{.}&space;p_{r}(x)\log&space;\widehat{y}\,&space;&plus;&space;p_{g}(x)log(1\,&space;-\,&space;\log(\widehat{y}))&space;dx" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\small&space;L_{d}^{r}=&space;E_{x\sim&space;p_{r}}\log&space;\widehat{y}\:&space;&plus;&space;E_{x\sim&space;p_{g}}\log&space;1-&space;\widehat{y}&space;=&space;\int_{x}^{.}&space;p_{r}(x)\log&space;\widehat{y}\,&space;&plus;&space;p_{g}(x)log(1\,&space;-\,&space;\log(\widehat{y}))&space;dx" title="\small L_{d}^{r}= E_{x\sim p_{r}}\log \widehat{y}\: + E_{x\sim p_{g}}\log 1- \widehat{y} = \int_{x}^{.} p_{r}(x)\log \widehat{y}\, + p_{g}(x)log(1\, -\, \log(\widehat{y})) dx" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\small&space;\frac{\mathrm{d}&space;L_{d}^{r}}{\mathrm{d}&space;\widehat{y}}&space;=&space;0" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\small&space;\frac{\mathrm{d}&space;L_{d}^{r}}{\mathrm{d}&space;\widehat{y}}&space;=&space;0" title="\small \frac{\mathrm{d} L_{d}^{r}}{\mathrm{d} \widehat{y}} = 0" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\small&space;\frac{P_{r}(x)}{\widehat{y}}&space;-&space;\frac{P_{g}(x)}{1-\widehat{y}}&space;=&space;0" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\small&space;\frac{P_{r}(x)}{\widehat{y}}&space;-&space;\frac{P_{g}(x)}{1-\widehat{y}}&space;=&space;0" title="\small \frac{P_{r}(x)}{\widehat{y}} - \frac{P_{g}(x)}{1-\widehat{y}} = 0" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\widehat{y}^{*}&space;=&space;\frac{P_{r}(x)}{P_{r}(x)&space;&plus;&space;P_{g}(x)}&space;\rightarrow&space;2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\widehat{y}^{*}&space;=&space;\frac{P_{r}(x)}{P_{r}(x)&space;&plus;&space;P_{g}(x)}&space;\rightarrow&space;2" title="\widehat{y}^{*} = \frac{P_{r}(x)}{P_{r}(x) + P_{g}(x)} \rightarrow 2" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\small&space;L_{d}^{r^{*}}&space;=&space;\int_{x}^{.}&space;p_{r}(x)\log&space;(\frac{P_{r}(x)}{P_{r}(x)&space;&plus;&space;P_{g}(x)}),&space;&plus;&space;p_{g}(x)log(1\,&space;-\,&space;\frac{P_{r}(x)}{P_{r}(x)&space;&plus;&space;P_{g}(x)})&space;dx" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\small&space;L_{d}^{r^{*}}&space;=&space;\int_{x}^{.}&space;p_{r}(x)\log&space;(\frac{P_{r}(x)}{P_{r}(x)&space;&plus;&space;P_{g}(x)}),&space;&plus;&space;p_{g}(x)log(1\,&space;-\,&space;\frac{P_{r}(x)}{P_{r}(x)&space;&plus;&space;P_{g}(x)})&space;dx" title="\small L_{d}^{r^{*}} = \int_{x}^{.} p_{r}(x)\log (\frac{P_{r}(x)}{P_{r}(x) + P_{g}(x)}), + p_{g}(x)log(1\, -\, \frac{P_{r}(x)}{P_{r}(x) + P_{g}(x)}) dx" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\small&space;L_{d}^{r^{*}}&space;=&space;L(G,&space;D^{*})&space;=&space;\int_{x}^{.}&space;p_{r}(x)\log&space;(\frac{P_{r}(x)}{P_{r}(x)&space;&plus;&space;P_{g}(x)}),&space;&plus;&space;p_{g}(x)log(&space;\frac{P_{g}(x)}{P_{r}(x)&space;&plus;&space;P_{g}(x)})&space;dx&space;\rightarrow&space;3" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\small&space;L_{d}^{r^{*}}&space;=&space;L(G,&space;D^{*})&space;=&space;\int_{x}^{.}&space;p_{r}(x)\log&space;(\frac{P_{r}(x)}{P_{r}(x)&space;&plus;&space;P_{g}(x)}),&space;&plus;&space;p_{g}(x)log(&space;\frac{P_{g}(x)}{P_{r}(x)&space;&plus;&space;P_{g}(x)})&space;dx&space;\rightarrow&space;3" title="\small L_{d}^{r^{*}} = L(G, D^{*}) = \int_{x}^{.} p_{r}(x)\log (\frac{P_{r}(x)}{P_{r}(x) + P_{g}(x)}), + p_{g}(x)log( \frac{P_{g}(x)}{P_{r}(x) + P_{g}(x)}) dx \rightarrow 3" /></a>

So when `y_hat` = `y_hat*`, the discriminator is at its minimum. At the end of the training, if `G` does a good job at approximating `P_r`, then `P_g` ~ `P_r`. 

<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\because&space;\,&space;P_{r}(x)&space;\sim&space;P_{g}(x),\,&space;\,&space;\,&space;\,&space;\,&space;\widehat{y}^{*}&space;=&space;1/2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\because&space;\,&space;P_{r}(x)&space;\sim&space;P_{g}(x),\,&space;\,&space;\,&space;\,&space;\,&space;\widehat{y}^{*}&space;=&space;1/2" title="\because \, P_{r}(x) \sim P_{g}(x),\, \, \, \, \, \widehat{y}^{*} = 1/2" /></a>

substituting it in equation `3` gives the optimal loss of the discriminator at the end of the training.

<a href="https://www.codecogs.com/eqnedit.php?latex=L(G^{*},&space;D^{*})&space;=&space;-2\log&space;2&space;\rightarrow&space;4" target="_blank"><img src="https://latex.codecogs.com/gif.latex?L(G^{*},&space;D^{*})&space;=&space;-2\log&space;2&space;\rightarrow&space;4" title="L(G^{*}, D^{*}) = -2\log 2 \rightarrow 4" /></a>

This is the cost is obtained when both `D` and `G` are perfectly optimized.

From the JS divergence equation, the JS divergence between `P_g` and `P_r` is

<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\small&space;D_{JS}(P_{r}&space;||&space;P_{g})&space;=&space;\frac{1}{2}\&space;D_{KL}(P_{r}&space;||&space;\frac{P_{r}&space;&plus;&space;P_{g}}{2})&space;&plus;&space;\frac{1}{2}\&space;D_{KL}(P_{g}&space;||&space;\frac{P_{r}&space;&plus;&space;P_{g}}{2})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\small&space;D_{JS}(P_{r}&space;||&space;P_{g})&space;=&space;\frac{1}{2}\&space;D_{KL}(P_{r}&space;||&space;\frac{P_{r}&space;&plus;&space;P_{g}}{2})&space;&plus;&space;\frac{1}{2}\&space;D_{KL}(P_{g}&space;||&space;\frac{P_{r}&space;&plus;&space;P_{g}}{2})" title="\small D_{JS}(P_{r} || P_{g}) = \frac{1}{2}\ D_{KL}(P_{r} || \frac{P_{r} + P_{g}}{2}) + \frac{1}{2}\ D_{KL}(P_{g} || \frac{P_{r} + P_{g}}{2})" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\small&space;2D_{JS}(P_{r}&space;||&space;P_{g})&space;=&space;\int_{x}^{.}&space;p_{r}(x)\log&space;(\frac{2P_{r}(x)}{P_{r}(x)&space;&plus;&space;P_{g}(x)}),&space;&plus;&space;\int_{x}^{.}p_{g}(x)log(&space;\frac{2P_{g}(x)}{P_{r}(x)&space;&plus;&space;P_{g}(x)})&space;dx" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\small&space;2D_{JS}(P_{r}&space;||&space;P_{g})&space;=&space;\int_{x}^{.}&space;p_{r}(x)\log&space;(\frac{2P_{r}(x)}{P_{r}(x)&space;&plus;&space;P_{g}(x)}),&space;&plus;&space;\int_{x}^{.}p_{g}(x)log(&space;\frac{2P_{g}(x)}{P_{r}(x)&space;&plus;&space;P_{g}(x)})&space;dx" title="\small 2D_{JS}(P_{r} || P_{g}) = \int_{x}^{.} p_{r}(x)\log (\frac{2P_{r}(x)}{P_{r}(x) + P_{g}(x)}), + \int_{x}^{.}p_{g}(x)log( \frac{2P_{g}(x)}{P_{r}(x) + P_{g}(x)}) dx" /></a>

From equation `3`,

<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\small&space;2D_{JS}(P_{r}&space;||&space;P_{g})&space;=&space;2\log&space;2\,&space;\,&space;&plus;\,&space;\,&space;L(G,&space;D^{*})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\small&space;2D_{JS}(P_{r}&space;||&space;P_{g})&space;=&space;2\log&space;2\,&space;\,&space;&plus;\,&space;\,&space;L(G,&space;D^{*})" title="\small 2D_{JS}(P_{r} || P_{g}) = 2\log 2\, \, +\, \, L(G, D^{*})" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\small&space;2D_{JS}(P_{r}&space;||&space;P_{g})&space;-&space;2\log&space;2\,&space;\,&space;=\,&space;\,&space;L(G,&space;D^{*})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\small&space;2D_{JS}(P_{r}&space;||&space;P_{g})&space;-&space;2\log&space;2\,&space;\,&space;=\,&space;\,&space;L(G,&space;D^{*})" title="\small 2D_{JS}(P_{r} || P_{g}) - 2\log 2\, \, =\, \, L(G, D^{*})" /></a>

The JS divergence is positive semidefinite. Hence, for the value of the above equation to be equal to the value calculated in `4`, the JS divergence should be `0` i.e., `P_g` = `P_r`. To conclude, when the `D` is at its best, `G` need to make `P_g` ~ `P_r` to reach the global optimum.


 
