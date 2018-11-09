import numpy as np
import time
import pandas as pd
from numpy.linalg import inv, det
from scipy.stats import matrix_normal, multivariate_normal, invwishart, wishart, f

n = 1000

alpha=4

mu = [1,2,3,4,5]

Sigma =[[1,0.5],[0.5,1]]

p = len(mu)

np.random.seed(123)

X = multivariate_normal.rvs(mu, Sigma, n)

mean_X = sum(X)/n

S_X = (X-mean_X).T.dot(X-mean_X)

# Plug-in generation of dataset

V = multivariate_normal.rvs(mean_X, S_X/(n-1), n)

# PPS generation of dataset

tilde_inverse_Sigma = wishart.rvs(n+alpha-p-2, inv(S_X))

tilde_Sigma = inv(tilde_inverse_Sigma)

tilde_mu = multivariate_normal.rvs(mean_X, tilde_Sigma/(n))

W = multivariate_normal.rvs(tilde_mu, tilde_Sigma,n)

# Estimators Plug-in

mean_V = sum(V)/n

S_star = (V-mean_V).T.dot(V-mean_V)

# Estimators Posterior Predictive Sampling

mean_W = sum(W)/n

S_bullet = (W-mean_W).T.dot(W-mean_W)