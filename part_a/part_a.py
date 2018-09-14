# Importing functions from folder with common functions for project 1
import sys
sys.path.append('../functions')
from functions import FrankeFunction, MSE, R2_Score, create_X

# Importing global packages
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed

fig = plt.figure()
ax = fig.gca(projection='3d')

fig2 = plt.figure()
ax2 = fig2.gca(projection='3d')

fig3 = plt.figure()
ax3 = fig3.gca(projection='3d')

#np.random.seed(79162)

# Make data.
N = 1000
x = np.linspace(0,1,N)#np.sort(np.random.uniform(0, 1, N))
y = np.linspace(0,1,N)#np.sort(np.random.uniform(0, 1, N))
x_mesh_, y_mesh_ = np.meshgrid(x,y)

# Add noise
z = FrankeFunction(x_mesh_, y_mesh_)
z_noise = z + np.random.normal(scale = 1, size = (N,N))


# Ravel the data to a vector
x_ = np.ravel(x_mesh_)
y_ = np.ravel(y_mesh_)
z_noise = np.ravel(z_noise)

# Regression
n = 5
l = int((n+1)*(n+2)/2)		# Number of elements in beta
X = create_X(x_, y_)


# Calculate the regression, using pinv in case of singular matrix
beta = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(z_noise)

# Create best-fit matrix for plotting
x_r = np.linspace(0,1,N)
y_r = np.linspace(0,1,N)
x_mesh, y_mesh = np.meshgrid(x,y)
X_r = create_X(x_mesh, y_mesh, mesh = True)

z_reg = (X_r.dot(beta)).reshape((N,N))

# Analytical matrix for comparison
z = FrankeFunction(x_mesh, y_mesh)

# Plot the surface.of the best fit
surf = ax.plot_surface(x_mesh, y_mesh, z_reg, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
ax.set_title("Regression")



# Plot the analtical solution
surf2 = ax2.plot_surface(x_mesh, y_mesh, z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)


# Customize the z axis.
ax2.set_zlim(-0.10, 1.40)
ax2.zaxis.set_major_locator(LinearLocator(10))
ax2.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig2.colorbar(surf, shrink=0.5, aspect=5)
ax2.set_title("Analytical solution")



# Plot the surface.of the noisy data
z_noise = z_noise.reshape((N,N))	# Reshape the data to a matrix
surf3 = ax3.plot_surface(x_mesh_, y_mesh_, z_noise, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)


# Customize the z axis.
ax3.set_zlim(-1.10, 2.40)
ax3.zaxis.set_major_locator(LinearLocator(10))
ax3.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig3.colorbar(surf, shrink=0.5, aspect=5)
ax3.set_title("Data with noise")

print("MSE: %.5f" %MSE(z, z_reg))
print("R2_Score: %.5f" %R2_Score(z, z_reg))

plt.show()
