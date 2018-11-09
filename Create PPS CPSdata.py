import numpy as np
import pandas as pd
import time
from numpy.linalg import inv
from scipy.stats import multivariate_normal, invwishart, wishart, matrix_normal

# Import select data

Ndata = pd.read_csv('C:\\Users\\ricar\\Dropbox (Moura)\\Confidentiality_Research_in_Python\\Ndatacps.csv')

# Define Y and X again see CPSdatalog.py

Y = Ndata[['HALMVAL', 'PROPTAX', 'HTOTVAL']].T
X = Ndata.drop(columns=['HALMVAL', 'PROPTAX', 'HTOTVAL']).T

# Define parameters for MLR

[n, m, p, alpha] = [len(Ndata), len(Y), len(X), 8]

# M defines how many multiple versions one needs

M = 5

# Make Betahat and S

R_invXXTX = inv(X.dot(X.T)).dot(X)  # Used several times

Beta_hat = R_invXXTX.dot(Y.T)
Beta_hat = pd.DataFrame(Beta_hat, index=X.index, columns=Y.index)
#  Flat_Beta_hat = Beta_hat.T.values.flatten()  # needed to create Btilde (not needed python as matrix_normal)

R_BetahatTX = Beta_hat.T.dot(X)  # Used several times and used for Beta_hat.T.dot(X[i])

S = (Y - R_BetahatTX).dot((Y - R_BetahatTX).T) / (n - p)
S = pd.DataFrame(S, index=Y.index, columns=Y.index)

# Draw multiple versions of Y: 1)the globals()['W%s' % i] creates the M matrices of zeros with the DataFrame of Y
for i in range(1, M + 1):
    globals()['W%s' % i] = Y.copy() * 0

# Draw multiple versions of Y: 2) use the globals()['W%s' % i] to automatically
#  create M plug-in sythetic versions  of Y and extract Bbull's and Sbull's

np.random.seed(1234)

Bsum0 = Beta_hat.copy() * 0
Ssum0 = S.copy() * 0
Wsum0 = Y.copy() * 0
S_np = S * (n - p)
S_bullcomb2 = Ssum0.copy()
Smean22 = Ssum0.copy()

start_time = time.time()

for k in range(0, 100):
    Bsumb = Bsum0.copy()
    Ssumb = Ssum0.copy()
    Wsum = Wsum0.copy()  # reset sums in each simulation
    for j in range(1, M + 1):
        globals()['TildeS%s' % j] = invwishart.rvs(df=n + alpha - p, scale=S_np)
        ## globals()['FlatTildeB%s' % j] = multivariate_normal.rvs(mean=Flat_Beta_hat,
        ##                                                       cov=np.kron(globals()['TildeS%s' % j], inv(X.dot(X.T))))
        globals()['TildeB%s' % j] = pd.DataFrame(matrix_normal.rvs(
            mean=Beta_hat, rowcov=inv(X.dot(X.T)), colcov=globals()['TildeS%s' % j])
            , index=X.index, columns=Y.index)  # matrix normal random variables

        globals()['TilBTX%s' % j] = globals()['TildeB%s' % j].T.dot(X)  # needed for next
        for i in range(0, n):
            globals()['W%s' % j][i] = multivariate_normal.rvs(mean=globals()['TilBTX%s' % j][i],
                                                              cov=globals()[
                                                                  'TildeS%s' % j])  # construct each PPS synthetic dataset
        # stopped checking here
        globals()['B_bull%s' % j] = R_invXXTX.dot(globals()['W%s' % j].T)  # create M B_bull's
        globals()['B_bull%s' % j] = pd.DataFrame(globals()['B_bull%s' % j], index=X.index, columns=Y.index)
        globals()['S_bull%s' % j] = (globals()['W%s' % j] - globals()['B_bull%s' % j].T.dot(X)) \
                                        .dot((globals()['W%s' % j] - globals()['B_bull%s' % j].T.dot(X)).T) / (n - p)
        globals()['S_bull%s' % j] = pd.DataFrame(globals()['S_bull%s' % j], index=Y.index, columns=Y.index)
        Wsum = Wsum + globals()['W%s' % j]  # add all M synthetic versions W's
        Bsumb = Bsumb + globals()['B_bull%s' % j]  # add all M synthetic versions B's
        Ssumb = Ssumb + globals()['S_bull%s' % j]  # add all M synthetic versions S's

    # Using second procedure

    Wmean = Wsum / M
    Bmeanb = Bsumb / M
    Smeanb = Ssumb / M

    Svb = Ssum0.copy()

    # for i in range(0, n):
    #     Rsamp = S.copy() * 0
    #     for j in range(1, M + 1):
    #         Temp = pd.DataFrame(globals()['W%s' % j][i] - Vmean[i])
    #         Rsamp = Rsamp + Temp.dot(Temp.T)
    #     Sv = Sv + Rsamp
    for j in range(1, M + 1):
        Tempb = pd.DataFrame(globals()['W%s' % j] - Wmean)
        Svb = Svb + Tempb.dot(Tempb.T)

    B_2bull = R_invXXTX.dot(Wmean.T)
    B_2bull = pd.DataFrame(B_2bull, index=X.index, columns=Y.index)
    S_2bull = (Wmean - Bmeanb.T.dot(X)).dot((Wmean - Bmeanb.T.dot(X)).T)
    S_2bull = pd.DataFrame(S_2bull, index=Y.index, columns=Y.index)
    S_bullcomb = (M * S_2bull + Svb) / (M * n - p)
    S_bullcomb2 = S_bullcomb2 + S_bullcomb
    Smean22 = Smean22 + Smeanb
end_time = time.time()
print("time:", end_time - start_time)
print(S_bullcomb2 / (k + 1) - S)
print(Smean22 / (k + 1) - S)

#  As one can see, the combination covariance is somewhat distant from the S, but would it be good to observer only Svb: it
#  it seems to be closer to S.


