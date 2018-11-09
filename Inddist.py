import numpy as np
from numpy.linalg import det
from scipy.stats import wishart
from partition import partition # function created to partition covariance matrices


def Inddist(part,nsample,pvariates,iterations):
    T = []
    for i in range(iterations):
        W1 = wishart.rvs(df = nsample-1,scale=np.eye(pvariates))
        W2 = wishart.rvs(df = nsample-1,scale=W1/(nsample-1))
        W2_11, W2_12, W2_21, W2_22 = partition(W2, part, part)
        T = np.append(T,det(W2)/(det(W2_11)*det(W2_22)))
    return T
