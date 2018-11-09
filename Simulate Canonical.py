import numpy as np
from partition import partition # function created to partition covariance matrices
from numpy.linalg import inv, det
from scipy.stats import multivariate_normal
import seaborn as sns
import time
import matplotlib.pyplot as plt
from Canodist import Canodist


np.random.seed(1234)
sim = 100000

#Sample size and partition size
n = 100
p1 = 2
p2 = 1

# Some population mean
mu = np.array([1,2,3,4])
# Number of covariates
p = len(mu)

# Three different population variances and respecive partitions

Sigma1 = np.array([[1,0.5,0.5,0.5],
          [0.5,1,0.5,0.5],
          [0.5,0.5,1,0.5],
          [0.5,0.5,0.5,1]])


Sigma1_11 , Sigma1_12 , Sigma1_21, Sigma1_22 = partition(Sigma1,p1,p1)

Sigma1_112 = Sigma1_11 - Sigma1_12.dot(inv(Sigma1_22).dot(Sigma1_21))

Delta1 = Sigma1_12.dot(inv(Sigma1_22))

Sigma2 = np.array([[1,0.5,0,0],
          [0.5,2,0,0],
          [0,0,3,0.2],
          [0,0,0.2,4]])




Sigma2_11 , Sigma2_12 , Sigma2_21, Sigma2_22 = partition(Sigma2,p2,p2)

Sigma2_112 = Sigma2_11 - Sigma2_12.dot(inv(Sigma2_22).dot(Sigma2_21))

Delta2 = Sigma2_12.dot(inv(Sigma2_22))

start_time = time.time()
T1 = Canodist(p1,n,p,sim)
T2 = Canodist(p2,n,p,sim)
end_time = time.time()
dtime = end_time - start_time
print("time1:", dtime)

q951 = np.percentile(T1,95)
q952 = np.percentile(T2,95)

T1_1 = []
T1_2 = []

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

    S_star1_11, S_star1_12, S_star1_21, S_star1_22 = partition(S_star1, p1, p1)
    S_star1_112 = S_star1_11 - S_star1_12.dot(inv(S_star1_22).dot(S_star1_21))

    Delta_star1 = S_star1_12.dot(inv(S_star1_22))

    S_star2_11, S_star2_12, S_star2_21, S_star2_22 = partition(S_star2, p2, p2)
    S_star2_112 = S_star2_11 - S_star2_12.dot(inv(S_star2_22).dot(S_star2_21))

    Delta_star2 = S_star2_12.dot(inv(S_star2_22))


    T1temp = det((Delta_star1-Delta1).dot(S_star1_22.dot((Delta_star1-Delta1).T)))/(det(S_star1_112))
    T2temp = det((Delta_star2-Delta2).dot(S_star2_22.dot((Delta_star2-Delta2).T)))/(det(S_star2_112))

    T1_1 = np.append(T1_1, T1temp)
    T1_2 = np.append(T1_2, T2temp)



end_time = time.time()
dtime = end_time - start_time
print("time2:", dtime)


print(min(T1_1),np.percentile(T1_1,10),np.percentile(T1_1,50),np.percentile(T1_1,90),max(T1_1))
print(min(T1_2),np.percentile(T1_2,10),np.percentile(T1_2,50),np.percentile(T1_2,90),max(T1_2))
print(min(T1),np.percentile(T1,10),np.percentile(T1,50),np.percentile(T1,90),max(T1))
print(min(T2),np.percentile(T2,10),np.percentile(T2,50),np.percentile(T2,90),max(T2))


plt.figure(1)
plt.subplot(321)
sns.kdeplot(T1, label='T1', cut=0,color='b')
plt.subplot(322)
sns.kdeplot(T2, label='T2', cut=0,color='b')
plt.subplot(323)
sns.kdeplot(T1_1, label='T1_1',cut=0,color='g')
plt.subplot(325)
sns.kdeplot(T1_2, label='T1_2',cut=0,color='y')
plt.legend()

plt.figure(3)
sns.kdeplot(T1_1**(1/p), label='T4_3',cut=0,color='g')
sns.kdeplot(T1**(1/p), label='T', cut=0,color='b')
plt.figure(4)
sns.kdeplot(T1_2**(1/p), label='T1_2',cut=0,color='y')
sns.kdeplot(T2**(1/p), label='T2', cut=0,color='b')
plt.legend()

cov1 = np.mean((T1_1<q951))
cov2 = np.mean((T1_2<q952))


print([cov1,cov2])
