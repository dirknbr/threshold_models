# Kink and jump threshold models (in stan)

This uses the Hidalgo (2018) threshold model. Imagine you have three variables
$x_1$, q and y. We are mainly interested how $x_1$ predicts y. But above a 
certain threshold in q the relationship between y and $x_1$ changes. 

We have the jump model

$$y_i \sim N(a + (b_1 + b_2 I(q_i > \lambda))x_{1i}, \sigma)$$

where I() is the indicator function.

And we have the kink model

$$y_i \sim N(a + (b_1 + b_2 I(q_i > \lambda)(q_i - \lambda))x_{1i}, \sigma)$$

Now since it might be hard to find the right threshold $\lambda$ we normalise
q via minmax scaling to be between [0, 1], which means we can then use a
beta prior for $\lambda$.

We use the WAIC metric to judge both models on some simulated data (N=50).
