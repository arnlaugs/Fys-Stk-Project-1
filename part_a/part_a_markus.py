# Importing functions from folder with common functions for project 1
import sys
sys.path.append('../functions')
from functions import FrankeFunction, MSE, R2_Score, create_X, plot_surfaces

# Importing global packages
import numpy as np


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

print("MSE: %.5f" %MSE(z, z_reg))
print("R2_Score: %.5f" %R2_Score(z, z_reg))

z_noise = z_noise.reshape((N,N))

plot_surface(x_mesh, y_mesh, z_reg, "Regression")
plot_surface(x_mesh, y_mesh, z, "Analytical")
plot_surface(x_mesh_, y_mesh_, z_noise, "Data with noise", show= True)
