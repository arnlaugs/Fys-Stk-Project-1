import sys
sys.path.append('../functions')

from functions import FrankeFunction, MSE, R2_Score, create_X, train_test_data

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed


def linear(X,X_learn,z):

    beta = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(z)
    zpredict_ = X_learn.dot(beta)
    #Reshape to a matrix
    return beta,zpredict_



n_x=100
# Make data.
#x = np.arange(0, 1, 0.05)
#x = np.linspace(0,1,n_x)
#y = np.linspace(0,1,n_x)

x = np.sort(np.random.uniform(0, 1, n_x))
y = np.sort(np.random.uniform(0, 1, n_x))
#y = np.arange(0, 1, 0.05)
x, y = np.meshgrid(x,y)
z = FrankeFunction(x, y)

#Transform from matricies to vectors
x_1=np.ravel(x)
y_1=np.ravel(y)
z_1=np.ravel(z)
n=int(len(x_1))


i=np.arange(n)

#Bootstrap
antall=int(n*0.1)
MSE_=1
R2_=0
for t in range(100):
    x_,y_,z_,x_test,y_test,z_test=train_test_data(x_1,y_1,z_1,np.random.choice(n,antall,replace=False))
    X= create_X(x_,y_)
    X_learn= create_X(x_test,y_test)
    beta,zpredict=linear(X,X_learn,z_)
    if MSE(z_test,zpredict) <MSE_:
        MSE_=MSE(z_test,zpredict)
        beta_MSE=beta
        l=t
    if R2_Score(z_test,zpredict)>R2_:
        R2_=R2_Score(z_test,zpredict)
        beta_R2=beta
        o=t
print(o,R2_,l,MSE_)
print(beta_MSE)
print(beta_R2)

#K-fold
o=0
l=0
k=10
n_k=int(n/k)
MSE_=1
R2_=0
np.random.shuffle(i)

print(i[int(2*n_k):int((2+1)*n_k)])
for t in range(k):
    x_,y_,z_,x_test,y_test,z_test=train_test_data(x_1,y_1,z_1,i[t*n_k:(t+1)*n_k])
    X= create_X(x_,y_)
    X_learn= create_X(x_test,y_test)
    beta,zpredict=linear(X,X_learn,z_)
    if MSE(z_test,zpredict) <MSE_:
        MSE_=MSE(z_test,zpredict)
        beta_MSE=beta
        l=t
    if R2_Score(z_test,zpredict)>R2_:
        R2_=R2_Score(z_test,zpredict)
        beta_R2=beta
        o=t
print(o,R2_,l,MSE_)
print(beta_MSE)
print(beta_R2)       

    



