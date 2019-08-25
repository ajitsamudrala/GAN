Intro to GANs
===

A GAN takes a random sample from a high dimensional distribution as input and maps it to the data space. The task of learning is to learn a high capacity deterministic function that can efficiently capture the dependencies and patterns in the data so that the mapped point resembles a sample generated from the data distribution. 

Example:

In the below example, I have generated 300 samples from Isotropic bivariate guassian distribution. 

![bi var guass](Images/bi_var_guassian.png)

When passed through a function <a href="https://www.codecogs.com/eqnedit.php?latex=f(x)&space;=&space;x/10&space;&plus;&space;x/||x||" target="_blank"><img src="https://latex.codecogs.com/svg.latex?f(x)&space;=&space;x/10&space;&plus;&space;x/||x||" title="f(x) = x/10 + x/||x||" /></a>, the points form a ring, which demonstrates that if we have a high capacity function, we can learn patterns in high dimensional data like images.

![ring](Images/Ring_formation.png)



