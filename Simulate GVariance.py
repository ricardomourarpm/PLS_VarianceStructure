import numpy as np
from partition import partition # function created to partition covariance matrices
from numpy.linalg import inv, det
from scipy.stats import multivariate_normal
import seaborn as sns
import time
import matplotlib.pyplot as plt
from GVdist import GVdist


np.random.seed(1234)
sim = 100000

#Sample size and partition size
n = 100

# Some population mean
mu = np.array([1,2,3,4])
# Number of covariates
p = len(mu)

# Three different population variances and respecive partitions

Sigma1 = np.array([[1,0.5,0.5,0.5],
          [0.5,1,0.5,0.5],
          [0.5,0.5,1,0.5],
          [0.5,0.5,0.5,1]])


Sigma2 = np.array([[1,0.5,0,0],
          [0.5,2,0,0],
          [0,0,3,0.2],
          [0,0,0.2,4]])




start_time = time.time()
T = GVdist(n,p,sim)
q975 = np.percentile(T,97.5)
q025 = np.percentile(T,2.5)
T1_1 = []
T1_2 = []
end_time = time.time()
dtime = end_time - start_time
print("time1:", dtime)

start_time = time.time()
for i in range(sim):

    # Generate original data sample from normal with mu and Sigma

    X1 = multivariate_normal.rvs(mu, Sigma1, n)
    X2 = multivariate_normal.rvs(mu, Sigma2, n)

    mean1 = sum(X1)/n
    mean2 = sum(X2)/n

    S1 = (X1-mean1).T.dot(X1-mean1)
    S2 = (X2-mean2).T.dot(X2-mean2)

    # Generate PLS synthetic single data

    V1 = multivariate_normal.rvs(mean1, S1/(n-1), n)
    V2 = multivariate_normal.rvs(mean2, S2/(n-1), n)

    #PLS estimates of mu and Sigma

    meanV1 = sum(V1)/n
    meanV2 = sum(V2)/n

    S_star1 = (V1 - meanV1).T.dot(V1 - meanV1)
    S_star2 = (V2 - meanV2).T.dot(V2 - meanV2)

    T1temp = (n - 1) ** p * det(S_star1) / det(Sigma1)
    T2temp = (n - 1) ** p * det(S_star2) / det(Sigma2)

    T1_1 = np.append(T1_1, T1temp)
    T1_2 = np.append(T1_2, T2temp)


end_time = time.time()
dtime = end_time - start_time
print("time2:", dtime)


print(min(T1_1),np.percentile(T1_1,10),np.percentile(T1_1,50),np.percentile(T1_1,90),max(T1_1))
print(min(T1_2),np.percentile(T1_2,10),np.percentile(T1_2,50),np.percentile(T1_2,90),max(T1_2))
print(min(T),np.percentile(T,10),np.percentile(T,50),np.percentile(T,90),max(T))


plt.figure(1)
plt.subplot(224)
sns.kdeplot(T**(1/p), label='T', cut=0,color='b')
plt.subplot(221)
sns.kdeplot(T1_1**(1/p), label='T1_1',cut=0,color='g')
plt.subplot(222)
sns.kdeplot(T1_2**(1/p), label='T1_2',cut=0,color='y')
plt.legend()

plt.figure(2)
sns.kdeplot(T, label='T', cut=0,color='b')
sns.kdeplot(T1_1, label='T1_3',cut=0,color='g')
sns.kdeplot(T1_2, label='T1_2',cut=0,color='y')
plt.legend()

cov1 = np.mean((T1_1<q975) & (T1_1>q025))
cov2 = np.mean((T1_2<q975) & (T1_2>q025))

print([cov1,cov2])
