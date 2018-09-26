import sys
sys.path.append('../functions')


from functions import *
sys.path.append('../part_b')
from ridge import Ridge


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





n_x=1000
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


print(K_fold(x,y,z,10,0.1, "OLS"))
print(Bootstrap(x_1,y_1,z_1,10,0.1,"Ridge"))

     

    



