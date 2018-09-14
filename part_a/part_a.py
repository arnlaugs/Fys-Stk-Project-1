import sys
sys.path.append('../functions')

from functions import FrankeFunction, MSE, R2_Score

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed


def linear(X,z):

    beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(z_)

    zpredict_ = X.dot(beta)
    #Reshape to a matrix
    return beta,np.reshape(zpredict_,z.shape)

def train_test_bootstrap(x_,y_,z_,i):
	#print (np.sort(i))
	x_learn=np.delete(x_,i)
	y_learn=np.delete(y_,i)
	z_learn=np.delete(z_,i)
	x_test=np.take(x_,i)
	y_test=np.take(y_,i)
	#print(x_learn.shape)
	#print(type(x_learn))
	return x_learn,y_learn,z_learn,x_test,y_test


n_x=30
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

#i=np.random.randint(n-1, size=int(n*0.2))
i=np.arange(n)
#i=np.random.shuffle(i)
antall=int(n*0.2)

for t in range(10):
	x_,y_,z_,x_test,y_test=train_test_bootstrap(x_1,y_1,z_1,np.random.choice(n,antall,replace=False))
	#print(x_.shape,y_.shape,x_test.shape,y_test.shape,np.ones((n-antall,1)).shape)
	X=np.c_[np.ones((n-antall,1)),x_,y_,x_*x_,y_*x_,y_*y_,x_**3,x_**2*y_,x_*y_**2,y_**3,
	        x_**4,x_**3*y_,x_**2*y_**2,x_*y_**3,y_**4,
	        x_**5,x_**4*y_,x_**3*y_**2,x_**2*y_**3,x_*y_**4,y_**5]

	#print(X.shape)
	beta,zpredict=linear(X,z_)

	MSE_=MSE(z_,zpredict)
	print(t, "R2=",R2_Score(z_,zpredict))
	print(t, "MSE=",MSE_)
#5th order








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
