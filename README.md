Intro to GANs
===

A GAN takes a random sample from a high dimensional distribution as input and maps it to the data space. The task of learning is to learn a high capacity deterministic function that can efficiently capture the dependencies and patterns in the data so that the mapped point resembles a sample generated from the data distribution. 

Example:
---

I have generated 300 samples from `Isotropic Bivariate Gaussian distribution`. 

![bi var guass](Images/bi_var_guassian.png)

When passed through a function <a href="https://www.codecogs.com/eqnedit.php?latex=f(x)&space;=&space;x/10&space;&plus;&space;x/||x||" target="_blank"><img src="https://latex.codecogs.com/svg.latex?f(x)&space;=&space;x/10&space;&plus;&space;x/||x||" title="f(x) = x/10 + x/||x||" /></a>, the points form a `ring`, which demonstrates that a high capacity function like neural network can learn patterns in high dimensional data like images.

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

DCGANs
----

As aforementioned, GANs take a random sample from the latent space as an input and maps it to data space. In DCGANs, the mapping function is a deep neaural network, which is differentiable and parameterized by network weights. The mapping function is called `Generator(G)`. A `Discriminator(D)` is also a deep neaural network that takes a sample in the data space and maps it to the action space i.e., the probability of that sample being generated from data distribution. 

<a href="https://www.codecogs.com/eqnedit.php?latex=P_{z}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?P_{z}" title="P_{z}" /></a> :  Prior / latent distribution. Typically, this space is much smaller than the data space.

<a href="https://www.codecogs.com/eqnedit.php?latex=P_{g}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?P_{g}" title="P_{g}" /></a>: Data distribution of data generated by the generator

<a href="https://www.codecogs.com/eqnedit.php?latex=P_{r}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?P_{r}" title="P_{r}" /></a>: Real data distribution

D__m = (d_1, d_2, d_3, ... d_m) be the data generated according to P_r

G__n = (g_m+1, g_m+2, .. g_n) be the data generated according to P_g


Train `D` to minimize the emperical loss. 

<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\fn_cm&space;\small&space;L_{d}&space;=&space;\min_{\theta&space;_{d}}-&space;\sum_{i=1}^{n}\1(x_{i}&space;\epsilon&space;D_{m})log&space;\widehat{y}_{i}&space;&plus;&space;1(x_{i}\epsilon&space;G_{n})&space;\log&space;(1-&space;\widehat{y}_{i}))" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\fn_cm&space;\small&space;L_{d}&space;=&space;\min_{\theta&space;_{d}}-&space;\sum_{i=1}^{n}\1(x_{i}&space;\epsilon&space;D_{m})log&space;\widehat{y}_{i}&space;&plus;&space;1(x_{i}\epsilon&space;G_{n})&space;\log&space;(1-&space;\widehat{y}_{i}))" title="\small L_{d} = \min_{\theta _{d}}- \sum_{i=1}^{n}\1(x_{i} \epsilon D_{m})log \widehat{y}_{i} + 1(x_{i}\epsilon G_{n}) \log (1- \widehat{y}_{i}))" /></a>

Fix the `D` network, and train `G` to maximize the loss of `D` over G_n. 

<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\fn_cm&space;\small&space;L_{g}&space;=&space;\max_{\theta&space;_{g}}&space;-&space;\sum_{i=1}^{n}\&space;1(x_{i}\epsilon&space;G_{n})&space;\log&space;(1-&space;\widehat{y}_{i}))&space;=&space;\min_{\theta&space;_{g}}&space;\sum_{i=1}^{n}\&space;1(x_{i}\epsilon&space;G_{n})&space;\log&space;(1-&space;\widehat{y}_{i}))" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\fn_cm&space;\small&space;L_{g}&space;=&space;\max_{\theta&space;_{g}}&space;-&space;\sum_{i=1}^{n}\&space;1(x_{i}\epsilon&space;G_{n})&space;\log&space;(1-&space;\widehat{y}_{i}))&space;=&space;\min_{\theta&space;_{g}}&space;\sum_{i=1}^{n}\&space;1(x_{i}\epsilon&space;G_{n})&space;\log&space;(1-&space;\widehat{y}_{i}))" title="\small L_{g} = \max_{\theta _{g}} - \sum_{i=1}^{n}\ 1(x_{i}\epsilon G_{n}) \log (1- \widehat{y}_{i})) = \min_{\theta _{g}} \sum_{i=1}^{n}\ 1(x_{i}\epsilon G_{n}) \log (1- \widehat{y}_{i}))" /></a>

As stated in the original paper, in the early training period, the above loss doesn't offer enough gradient to update the parameters of `G` network, as initially `P_g` is distant from `P_d`, which makes it easy for `D` to classify generated images. Hence, we try to maximize it my switching labels.

<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\fn_cm&space;\small&space;L_{g}&space;=\min_{\theta&space;_{g}}&space;\sum_{i=1}^{n}\&space;1(x_{i}\epsilon&space;G_{n})&space;\log&space;(1-&space;\widehat{y}_{i}))&space;=&space;\max_{\theta&space;_{g}}&space;\sum_{i=1}^{n}\&space;1(x_{i}\epsilon&space;G_{n})&space;\log&space;(\widehat{y}_{i}))&space;=&space;\min_{\theta&space;_{g}}&space;-&space;\sum_{i=1}^{n}\&space;1(x_{i}\epsilon&space;G_{n})&space;\log&space;(\widehat{y}_{i})" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\inline&space;\fn_cm&space;\small&space;L_{g}&space;=\min_{\theta&space;_{g}}&space;\sum_{i=1}^{n}\&space;1(x_{i}\epsilon&space;G_{n})&space;\log&space;(1-&space;\widehat{y}_{i}))&space;=&space;\max_{\theta&space;_{g}}&space;\sum_{i=1}^{n}\&space;1(x_{i}\epsilon&space;G_{n})&space;\log&space;(\widehat{y}_{i}))&space;=&space;\min_{\theta&space;_{g}}&space;-&space;\sum_{i=1}^{n}\&space;1(x_{i}\epsilon&space;G_{n})&space;\log&space;(\widehat{y}_{i})" title="\small L_{g} =\min_{\theta _{g}} \sum_{i=1}^{n}\ 1(x_{i}\epsilon G_{n}) \log (1- \widehat{y}_{i})) = \max_{\theta _{g}} \sum_{i=1}^{n}\ 1(x_{i}\epsilon G_{n}) \log (\widehat{y}_{i})) = \min_{\theta _{g}} - \sum_{i=1}^{n}\ 1(x_{i}\epsilon G_{n}) \log (\widehat{y}_{i})" /></a>



 
