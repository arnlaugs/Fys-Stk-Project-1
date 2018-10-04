# Importing functions from folder with common functions for project 1
import sys
sys.path.append('../functions')
sys.path.append('../part_a')      # Add folder with OLS
sys.path.append('../part_b')      # Add folder with Ridge
from functions import *
from regression import OLS, Ridge, Lasso
#from sklearn.linear_model import Lasso

import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm



# Load the terrain
terrain = imread('n59_e010_1arc_v3.tif')
print(np.shape(terrain))
N = 1800
terrain = terrain[terrain.shape[0]-N-1800:1801,terrain.shape[1]-N:]
print(terrain.shape)
m=10
# Creates mesh of image pixels
x = np.linspace(0,1, np.shape(terrain)[0])
y = np.linspace(0,1, np.shape(terrain)[1])
x_mesh, y_mesh = np.meshgrid(x,y)

X = create_X(x_mesh, y_mesh,n=m)

R_ols = OLS();                  R_ols.fit(X, terrain)
#R_r = Ridge(lmbda=1.0);         R_r.fit(X, terrain)
#R_l = Lasso(alpha=0.1, fit_intercept=False); R_l.fit(X, np.ravel(terrain))



# Plotting surfaces and printing the MSE and R2-score
labels = ['Ordinary least squares', 'Ridge', 'Lasso']
i=0

for method in [R_ols]:
    z_reg = (method.predict(X)).reshape((N,N))
    fig, ax ,surf=plot_surface(x_mesh, y_mesh, (z_reg.reshape((N,N)).T), labels[i] + " m= "+str(m)+ " N=500", show=True,cmap=cm.viridis)

    print('============================')
    print(labels[i])
    print("MSE: %.5f" %MSE(terrain, z_reg))
    print("R2-score: %.5f \n" %R2_Score(terrain, z_reg))

    fig.show()
    #fig.savefig(labels[i] + str(m)+ "N_500"+".png")
    i+=1


print(x_mesh.shape, y_mesh.shape, terrain.shape)
# Plots surface plot

fig, ax ,surf = plot_surface(x_mesh, y_mesh, terrain.T,"Terrain, N=1800", cmap=cm.viridis)
#fig.savefig("Terrain_N_1800.png")
#ax.view_init(azim=10,elev=45)
plt.show()

# Shows image
"""
plt.figure()
plt.title('Terrain over the Oslo area')
plt.imshow(terrain, cmap='gray')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
"""
