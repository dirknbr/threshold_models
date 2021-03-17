
import numpy as np
import pystan

def waic(ll):
  lppd = np.log(np.exp(ll).mean(axis=0)).sum()
  p_waic = np.var(ll, axis=0).sum()
  return -2 * lppd + 2 * p_waic

N = 50
x1 = np.random.normal(10, 2, size=N)
q = np.random.normal(10, 2, size=N)
y = 5 + (1 + .6 * (q > 10.5) + .2 * (q > 11.5)) * x1 + np.random.normal(0, 1.5, size=N)

def norm(x):
  return (x - x.min()) / (x.max() - x.min())

qnorm = norm(q)

print(q, qnorm)

data = {'N': N, 'x1': x1, 'q': qnorm, 'y': y}

jump = pystan.StanModel('jump.stan')
jump_fit = jump.sampling(data, iter=2000, chains=2)
print(jump_fit)
jump_ll = jump_fit.extract()['log_lik']
print(waic(jump_ll))

kink = pystan.StanModel('kink.stan')
kink_fit = jump.sampling(data, iter=2000, chains=2)
print(kink_fit)
kink_ll = kink_fit.extract()['log_lik']
print(waic(kink_ll))
