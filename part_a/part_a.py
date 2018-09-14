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



n_x=1000
# Make data.
#x = np.arange(0, 1, 0.05)
x = np.linspace(0,1,n_x)
y = np.linspace(0,1,n_x)
#y = np.arange(0, 1, 0.05)
x, y = np.meshgrid(x,y)
z = FrankeFunction(x, y)

#Transform from matricies to vectors
x_1=np.ravel(x)
y_1=np.ravel(y)
z_1=np.ravel(z)
n=int(len(x_1))


i=np.arange(n)

antall=int(n*0.1)
MSE_=1
R2_=0
for t in range(10):
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
        k=t
print(k,R2_,l,MSE_)
print(beta_MSE)
print(beta_R2)








"""

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(x,y,zpredict)#,cmap=cm.coolwarm)



# Plot the surface.
surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
#fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
"""
