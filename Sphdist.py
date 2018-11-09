import numpy as np
from numpy.linalg import det
from scipy.stats import wishart

def Sphdist(nsample,pvariates,iterations):
    T = []
    for i in range(iterations):
        W1 = wishart.rvs(df = nsample-1,scale=np.eye(pvariates)/(nsample-1))
        W2 = wishart.rvs(df = nsample-1,scale=np.eye(pvariates))
        T = np.append(T,det(W1.dot(W2))**(1/pvariates)/np.trace(W1.dot(W2)))
    return T
