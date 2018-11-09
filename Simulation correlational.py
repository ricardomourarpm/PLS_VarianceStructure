import numpy as np
import time
import pandas as pd
from numpy.linalg import inv, det
from scipy.stats import matrix_normal, multivariate_normal, invwishart, wishart, f,chi2
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
import seaborn as sns
from partition import partition

n=100
p1=2


Sigma1 = pd.DataFrame([[1,0.5,0.5,0.5,0.5],
          [0.5,1,0.5,0.5,0.5],
          [0.5,0.5,1,0.4,0.4],
          [0.5,0.5,0.4,1,0.4],
          [0.5,0.5,0.4,0.4,1]])

p=len(Sigma1)

Sigma1_11 , Sigma1_12 , Sigma1_21, Sigma1_22 = partition(Sigma1,p1,p1)

Sigma1_112 = Sigma1_11 - Sigma1_12.dot(inv(Sigma1_22).dot(Sigma1_21))

Sigma2 = pd.DataFrame([[1,1,0.4,0.4,0.4],
          [1,2,0,0,0],
          [0.4,0,3,0.5,0.9],
          [0.4,0,0.5,4,0.9],
          [0.4,0,0.9,0.9,5]])




Sigma2_11 , Sigma2_12 , Sigma2_21, Sigma2_22 = partition(Sigma2,p1,p1)

Sigma2_112 = Sigma2_11 - Sigma2_12.dot(inv(Sigma2_22).dot(Sigma2_21))


Sigma3= pd.DataFrame([[1000, 0.5, 0  , 0  , 0.9],
                      [0.5 , 2  , 0  , 0  , 0],
                      [0   , 0  , 2  , 0.2, 0.7],
                      [0   , 0  , 0.2, 2  , 0.7],
                      [0.9   , 0  , 0.7, 0.7, 2]])



Sigma3_11 , Sigma3_12 , Sigma3_21, Sigma3_22 = partition(Sigma3,p1,p1)

Sigma3_112 = Sigma3_11 - Sigma3_12.dot(inv(Sigma3_22).dot(Sigma3_21))


T = []
T1 = []
T2 = []
T3 = []
T4 = []

Ex1 = []
Ex2 = []

dSigma1 = det(Sigma1)
dSigma2 = det(Sigma2)
dSigma3 = det(Sigma3)


for k in range(100):
    S1 = pd.DataFrame(wishart.rvs(df = n-1, scale = Sigma1))
    S1_star = pd.DataFrame(wishart.rvs(df = n-1, scale = S1/(n-1)))
    S2 = pd.DataFrame(wishart.rvs(df = n-1, scale = Sigma2))
    S2_star = pd.DataFrame(wishart.rvs(df = n-1, scale = S2/(n-1)))
    S3 = pd.DataFrame(wishart.rvs(df = n-1, scale = Sigma3))
    S3_star = pd.DataFrame(wishart.rvs(df = n-1, scale = S3/(n-1)))

    S1_11 = S1.iloc[0:2,0:2]
    S1_12 = S1.iloc[0:2, 2:]
    S1_21 = S1.iloc[2:, 0:2]
    S1_22 = S1.iloc[2:, 2:]

    S1_112 = S1_11-S1_12.dot(inv(S1_22).dot(S1_21))

    S1_star_11 = S1_star.iloc[0:2,0:2]
    S1_star_12 = S1_star.iloc[0:2, 2:]
    S1_star_21 = S1_star.iloc[2:, 0:2]
    S1_star_22 = S1_star.iloc[2:, 2:]

    S1_star_112 = S1_star_11-S1_star_12.dot(inv(S1_star_22).dot(S1_star_21))

    S2_11 = S2.iloc[0:2, 0:2]
    S2_12 = S2.iloc[0:2, 2:]
    S2_21 = S2.iloc[2:, 0:2]
    S2_22 = S2.iloc[2:, 2:]

    S2_112 = S2_11 - S2_12.dot(inv(S2_22).dot(S2_21))


    S2_star_11 = S2_star.iloc[0:2, 0:2]
    S2_star_12 = S2_star.iloc[0:2, 2:]
    S2_star_21 = S2_star.iloc[2:, 0:2]
    S2_star_22 = S2_star.iloc[2:, 2:]


    S2_star_112 = S2_star_11-S2_star_12.dot(inv(S2_star_22).dot(S2_star_21))

    S3_star_11 = S3_star.iloc[0:p1, 0:p1]
    S3_star_12 = S3_star.iloc[0:p1, p1:]
    S3_star_21 = S3_star.iloc[p1:, 0:p1]
    S3_star_22 = S3_star.iloc[p1:, p1:]

    S3_star_112 = S3_star_11-S3_star_12.dot(inv(S3_star_22).dot(S3_star_21))



    Wo = S3
    Wo_11 = Wo.iloc[0:2, 0:2]
    Wo_22 = Wo.iloc[2:, 2:]
    Wo_12 = Wo.iloc[0:2, 2:]
    Wo_21 = Wo.iloc[2:, 0:2]

    Wo_112 = Wo_11 - Wo_12.dot(inv(Wo_22).dot(Wo_21))

    W = pd.DataFrame(wishart.rvs(df=n - 1, scale=Wo / (n - 1)))
    W_11 = W.iloc[0:2,0:2]
    W_22 = W.iloc[2:, 2:]
    W_12 = W.iloc[0:2, 2:]
    W_21 = W.iloc[2:, 0:2]

    W_112 = pd.DataFrame(wishart.rvs(df=n-1-p+p1, scale=Wo_112/(n-1)))
            #W_11 - W_12.dot(inv(W_22).dot(W_21))

    Wish=wishart.rvs(df=n-1, scale=wishart.rvs(df=n-1,scale=np.identity(p))/(n-1))
        #wishart.rvs(df=n-1-p+p1, scale=np.identity(p1)/(n-1))
        #

    Dwstar=Wish[:p1,p1:].dot(inv(Wish[p1:p,p1:p]))


    D=Sigma1_12.dot(inv(Sigma1_22))
    Dhat=S1_12.dot(inv(S1_22))
    Dstar=S1_star_12.dot(inv(S1_star_22))

    D2 = Sigma2_12.dot(inv(Sigma2_22))
    Dhat2 = S2_12.dot(inv(S2_22))
    Dstar2 = S2_star_12.dot(inv(S2_star_22))

    mean = D*0
    variance = np.kron(inv(sqrtm(S1_112)).dot(Sigma1_112.dot(inv(sqrtm(S1_112)))),np.identity(p-p1))
    #inv(sqrtm(S1_112)).dot(Sigma1_112.dot(inv(sqrtm(S1_112))))
    #np.kron(inv(sqrtm(W_22)).dot((Wo_112/(n-1)).dot(inv(sqrtm(W_22)))), inv(sqrtm(Wo_112)).dot(W_22.dot(inv(sqrtm(Wo_112)))))
    #np.kron(np.identity(p1)/(n-1), np.identity(p-p1))

    meanflat = mean.flatten('F')

    Normal = multivariate_normal.rvs(mean= meanflat, cov = variance).reshape(mean.shape, order='F')

    Ex1temp=sum(sum(inv(sqrtm(S2_112)).dot((Dhat2-D2).dot(sqrtm(S2_22)))))
    Ex2temp=sum(sum(Normal))

    Ex1= np.append(Ex1,Ex1temp)
    Ex2 = np.append(Ex2, Ex2temp)

    Ttemp1 = det((Dstar-D).dot(S1_star_22.values.dot((Dstar-D).T)))/det(S1_star_112)
    Ttemp2 = det((Dstar2-D2).dot(S2_star_22.values.dot((Dstar2-D2).T)))/det(S2_star_112)
    #det(multivariate_normal.rvs(meanflat, variance,1).reshape(mean.shape,order='F'))
    Ttemp3 = det(S3_star_112)/det(S3_star_11)
    Ttemp4 = det(Dwstar.dot(Wish[p1:p,p1:p].dot(Dwstar.T)))/det(Wish[:p1,:p1]-Wish[:p1,p1:p].dot(inv(Wish[p1:p,p1:p]).dot(Wish[p1:p,:p1])))
    #Ttemp4 = det(Wish)/det(Wish+inv(sqrtm(Sigma3.iloc[0:2,0:2])).dot(W_12.dot(inv(W_22).dot(W_21).dot(inv(sqrtm(Sigma3.iloc[0:2,0:2]))))))


    T1 = np.append(T1,Ttemp1)
    T2 = np.append(T2, Ttemp2)
    T3 = np.append(T3, Ttemp3)
    T4 = np.append(T4, Ttemp4)


print(min(T1),np.percentile(T1,10),np.percentile(T1,50),np.percentile(T1,90),max(T1))
print(min(T2),np.percentile(T2,10),np.percentile(T2,50),np.percentile(T2,90),max(T2))
print(min(T3),np.percentile(T3,10),np.percentile(T3,50),np.percentile(T3,90),max(T3))
print(min(T4),np.percentile(T4,10),np.percentile(T4,50),np.percentile(T4,90),max(T4))


# plt.hist([T,T2],bins=25)

#  plt.hist(T2,bins=100)



sns.kdeplot(T1, label='T1', cut=0,color='r')
sns.kdeplot(T2, label='T2',cut=0,color='g')
sns.kdeplot(T3, label='T3',cut=0,color='y')
sns.kdeplot(T4, label='T4',cut=0,color='b')

sns.kdeplot(Ex1, label='T3',cut=0,color='y')
sns.kdeplot(Ex2, label='T4',cut=0,color='b')

plt.legend()


print(min(Ex1),np.percentile(Ex1,10),np.percentile(Ex1,50),np.percentile(Ex1,90),max(Ex1))
print(min(Ex2),np.percentile(Ex2,10),np.percentile(Ex2,50),np.percentile(Ex2,90),max(Ex2))