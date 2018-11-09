from chiprod import chiprod
import numpy as np

def GVdist(nsample,pvariates,iterations):
    T = []
    for i in range(iterations):
        q = chiprod(pvariates,nsample-1)
        p = chiprod(pvariates,nsample-1)
        T = np.append(T,p*q)
    return T
