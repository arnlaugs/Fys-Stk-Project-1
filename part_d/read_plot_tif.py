# Importing functions from folder with common functions for project 1
import sys
sys.path.append('../functions')
sys.path.append('../part_a')	  # Add folder with OLS
sys.path.append('../part_b')	  # Add folder with Ridge
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

N = 500
terrain = terrain[:N,:N]
m = 80
# Creates mesh of image pixels
x = np.linspace(0,1, np.shape(terrain)[0])
y = np.linspace(0,1, np.shape(terrain)[1])
x_mesh, y_mesh = np.meshgrid(x,y)


X = create_X(x_mesh, y_mesh,n=m)

R_ols = OLS()
#R_ols.fit(X, terrain)

#R_r = Ridge(lmbda=1.0)


#R_l = Lasso(alpha=0.1, fit_intercept=False);



# Plotting surfaces and printing the MSE and R2-score
labels = ['Ordinary Least Squares', 'Ridge', 'Lasso']

half_size = (2.64429, 1.98322)

i=0

for method in [R_ols]:
	method.fit(X, terrain)
	z_reg = (method.predict(X)).reshape((N,N))
	fig, ax , surf =plot_surface(x_mesh, y_mesh, (z_reg.reshape((N,N)).T), "", cmap=cm.viridis, figsize = half_size)

	print('============================')
	print(labels[i])
	print("MSE: %.5f" %MSE(terrain, z_reg))
	print("R2-score: %.5f \n" %R2_Score(terrain, z_reg))


	fig.savefig(("%s%iN%i.png" %(labels[i], m, N)).replace(" ", "_"), dpi = 200)
	#plt.show()
	#savefigure(labels[i] + "%iN%i" %(m, N), figure = fig)
	i+=1


#print(x_mesh.shape, y_mesh.shape, terrain.shape)
# Plots surface plot


fig2, ax2, surf2 = plot_surface(x_mesh, y_mesh, terrain.T, "", cmap=cm.viridis, figsize = half_size)
#plt.show()
fig2.savefig("terrainN%i.png" %(N), dpi = 200)
#fig.savefig("Terrain_N_1800.png")
#ax.view_init(azim=10,elev=45)
plt.show()
#savefigure("terrainN%i" %(N), figure = fig)

# Shows image
"""
plt.figure()
plt.title('Terrain over the Oslo area')
plt.imshow(terrain, cmap='gray')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
"""
