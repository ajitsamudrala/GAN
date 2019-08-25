Intro to GANs
===

A GAN takes a random sample from a high dimensional distribution as input and maps it to the data space. The task of learning is to learn a high capacity deterministic function that can efficiently capture the dependencies and patterns in the data so that the mapped point resembles a sample generated from the data distribution. 

Example:
---

I have generated 300 samples from `Isotropic Bivariate Gaussian distribution`. 

![bi var guass](Images/bi_var_guassian.png)

When passed through a function <a href="https://www.codecogs.com/eqnedit.php?latex=f(x)&space;=&space;x/10&space;&plus;&space;x/||x||" target="_blank"><img src="https://latex.codecogs.com/svg.latex?f(x)&space;=&space;x/10&space;&plus;&space;x/||x||" title="f(x) = x/10 + x/||x||" /></a>, the points form a `ring`, which demonstrates that if we have a high capacity function, we can learn patterns in high dimensional data like images.

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

<h3> Joint Distribution </h3>

A join distribution a.k.a data distribution captures the joint probabilities between random variables. In the above table, the join probability of `P(X = x_2 & Y = y_3)` is `2/50`. This is what a GAN tries to model from the sample data. 

Consider images of size `28 x 28`. Each pixel is a random variable that can take any value from 0 to 255. Hence, we have 784 random variables in total. GAN tries to model the dependencies between the pixels. 

<h3> Bayes Theorm </h3>

<a href="https://www.codecogs.com/eqnedit.php?latex=P(&space;A&space;|&space;B)&space;=&space;\frac{P(A&space;\bigcap&space;B)}{P(B))}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?P(&space;A&space;|&space;B)&space;=&space;\frac{P(A&space;\bigcap&space;B)}{P(B))}" title="P( A | B) = \frac{P(A \bigcap B)}{P(B))}" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=Conditional&space;Probability&space;=&space;\frac{Joint&space;Probability}{Marginal&space;Probability}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?Conditional&space;Probability&space;=&space;\frac{Joint&space;Probability}{Marginal&space;Probability}" title="Conditional Probability = \frac{Joint Probability}{Marginal Probability}" /></a>

From the above table: `P(Y = y_1 | X = x_1) = P(Y = y_1 & X = x_1) / P(X = x_1)` = `(2/50)/(17/50)` = `2/17`

<h3> Entropy </h3>

Entropy measures the degree of uncertainity of an outcome of a trail according to a `p(x)`. 

<a href="https://www.codecogs.com/eqnedit.php?latex=H(p)&space;=&space;-\sum_{k=1}^{K}p_{i}\log&space;p_{i}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?H(p)&space;=&space;-\sum_{k=1}^{K}p_{i}\log&space;p_{i}" title="H(p) = -\sum_{k=1}^{K}p_{i}\log p_{i}" /></a>

![entropy](Images/entropy.png)

 
The entropy of a unbiased coin is higher than biased coin. The difference in entropy increases with degree of ploraization of probabilites of biased coin.

<h3>Cross Entropy</h3>

Cross entropy measure degree of uncertainity of a trial according to `p(y)` but in truth according to `p(x)`.

<a href="https://www.codecogs.com/eqnedit.php?latex=H(p(x),&space;p(y))&space;=&space;-\sum_{k=1}^{K}p(x_{i})\log&space;p(y_{i})" target="_blank"><img src="https://latex.codecogs.com/svg.latex?H(p(x),&space;p(y))&space;=&space;-\sum_{k=1}^{K}p(x_{i})\log&space;p(y_{i})" title="H(p(x), p(y)) = -\sum_{k=1}^{K}p(x_{i})\log p(y_{i})" /></a>

![entropy](Images/cross_entropy.png)

Cross entropy is higher when a trial is conducted according to unbiased coin probability distribution but you think it is being conducted according to biased coin probability distribution. 


