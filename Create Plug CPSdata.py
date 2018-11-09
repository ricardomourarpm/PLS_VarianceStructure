import numpy as np
import pandas as pd
import time
from numpy.linalg import inv
from scipy.stats import multivariate_normal

# Import select data

Ndata = pd.read_csv('C:\\Users\\ricar\\Dropbox (Moura)\\Confidentiality_Research_in_Python\\Ndatacps.csv')

# Define Y and X again see CPSdatalog.py

Y = Ndata[['HALMVAL', 'PROPTAX', 'HTOTVAL']].T
X = Ndata.drop(columns=['HALMVAL', 'PROPTAX', 'HTOTVAL']).T

# Define parameters for MLR

[n, m, p] = [len(Ndata), len(Y), len(X)]

# M defines how many multiple versions one needs

M = 2

# Make Betahat and S

R_invXXTX = inv(X.dot(X.T)).dot(X)  # Used several times

Beta_hat = R_invXXTX.dot(Y.T)
Beta_hat = pd.DataFrame(Beta_hat, index=X.index, columns=Y.index)

R_BetahatTX = Beta_hat.T.dot(X)  # Used several times and used for Beta_hat.T.dot(X[i])

S = (Y - R_BetahatTX).dot((Y - R_BetahatTX).T) / (n - p)
S = pd.DataFrame(S, index=Y.index, columns=Y.index)

# Draw multiple versions of Y: 1)the globals()['V%s' % i] creates the M matrices of zeros with the DataFrame of Y
for i in range(1, M + 1):
    globals()['V%s' % i] = Y.copy() * 0

# Draw multiple versions of Y: 2) use the globals()['V%s' % i] to automatically
#  create M plug-in sythetic versions  of Y and extract Bstar's and Sstar's

np.random.seed(123)

Bsum0 = Beta_hat.copy() * 0
Ssum0 = S.copy() * 0
Vsum0 = Y.copy() * 0
S_comb2 = Ssum0.copy()

start_time = time.time()

for k in range(0, 200):
    Bsum = Bsum0.copy()
    Ssum = Ssum0.copy()
    Vsum = Vsum0.copy()  # reset sums in each simulation
    for j in range(1, M + 1):
        for i in range(0, n):
            globals()['V%s' % j][i] = multivariate_normal.rvs(mean=R_BetahatTX[i],
                                                              cov=S)  # construct each plug-in synthetic dataset
        globals()['B_star%s' % j] = R_invXXTX.dot(globals()['V%s' % j].T)  # create M B_star's
        globals()['B_star%s' % j] = pd.DataFrame(globals()['B_star%s' % j], index=X.index, columns=Y.index)
        globals()['S_star%s' % j] = (globals()['V%s' % j] - globals()['B_star%s' % j].T.dot(X)) \
                                        .dot((globals()['V%s' % j] - globals()['B_star%s' % j].T.dot(X)).T) / (n - p)
        globals()['S_star%s' % j] = pd.DataFrame(globals()['S_star%s' % j], index=Y.index, columns=Y.index)
        Vsum = Vsum + globals()['V%s' % j]  # add all M synthetic versions V's
        Bsum = Bsum + globals()['B_star%s' % j]  # add all M synthetic versions B's
        Ssum = Ssum + globals()['S_star%s' % j]  # add all M synthetic versions S's

    # Using second procedure

    Vmean = Vsum / M
    Bmean = Bsum / M
    Smean = Ssum / M

    Sv = Ssum0.copy()

    # for i in range(0, n):
    #     Rsamp = Ssum0
    #     for j in range(1, M + 1):
    #         Temp = pd.DataFrame(globals()['V%s' % j][i] - Vmean[i])
    #         Rsamp = Rsamp + Temp.dot(Temp.T)
    #     Sv = Sv + Rsamp replaced by: one less cycle
    for j in range(1, M + 1):
        Temp = pd.DataFrame(globals()['V%s' % j] - Vmean)
        Sv = Sv + Temp.dot(Temp.T)

    B_2star = R_invXXTX.dot(Vmean.T)
    B_2star = pd.DataFrame(B_2star, index=X.index, columns=Y.index)
    S_2star = (Vmean - B_2star.T.dot(X)).dot((Vmean - B_2star.T.dot(X)).T)
    S_2star = pd.DataFrame(S_2star, index=Y.index, columns=Y.index)
    S_comb = (M * S_2star + Sv) / (M * n - p)
    S_comb2 = S_comb2 + S_comb
end_time = time.time()
print("time:", end_time - start_time)
print(S_comb2 / (k + 1) - S)
