data {
  int N;
  vector[N] y;
  vector[N] x1;
  vector[N] q; // normalised to [0, 1]
}

parameters {
  real<lower=0, upper=1> lambda;
  real<lower=0> sigma;
  vector[2] b;
  real a;
}

transformed parameters {
  vector[N] mu;
  for (i in 1:N) mu[i] = a + (b[1] + b[2] * step(q[i] - lambda)) * x1[i];
}

model {
  lambda ~ beta(1, 1);
  sigma ~ gamma(1, 1);
  b ~ normal(0, 1);
  y ~ normal(mu, sigma);
}

generated quantities {
  vector[N] log_lik;
  for (i in 1:N) log_lik[i] = normal_lpdf(y[i] | mu[i], sigma);
}
