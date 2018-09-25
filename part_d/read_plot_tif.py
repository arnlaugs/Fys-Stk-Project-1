# Importing functions from folder with common functions for project 1
import sys
sys.path.append('../functions')
from functions import FrankeFunction, MSE, R2_Score, create_X, plot_surface, calc_beta
import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


# Load the terrain
terrain = imread('n59_e010_1arc_v3.tif')
print(np.shape(terrain))
terrain = terrain[:1800,:1800]

# Creates mesh of image pixels
x = np.linspace(0,1, np.shape(terrain)[0])
y = np.linspace(0,1, np.shape(terrain)[1])
x_mesh, y_mesh = np.meshgrid(x,y)


# Plots surface plot
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(x_mesh, y_mesh, terrain.T, cmap=cm.viridis,
                   linewidth=0, antialiased=False)

# Shows image
plt.figure()
plt.title('Terrain over the Oslo area')
plt.imshow(terrain, cmap='gray')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
