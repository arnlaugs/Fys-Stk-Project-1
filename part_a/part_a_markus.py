# Importing functions from folder with common functions for project 1
import sys
sys.path.append('../functions')
from functions import FrankeFunction, MSE, R2_Score, create_X, plot_surface

# Importing global packages
import numpy as np


#np.random.seed(79162)

# Make data.
N = 1000
x = np.sort(np.random.uniform(0, 1, N))
y = np.sort(np.random.uniform(0, 1, N))
x_mesh_, y_mesh_ = np.meshgrid(x,y)

# Add noise
z_noise = FrankeFunction(x_mesh_, y_mesh_) + np.random.normal(scale = 1, size = (N,N))


# Regression
X = create_X(x_mesh_, y_mesh_, mesh = True)


# Calculate the regression, using pinv in case of singular matrix
beta = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(np.ravel(z_noise))

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

plot_surface(x_mesh, y_mesh, z_reg, "Regression")
plot_surface(x_mesh, y_mesh, z, "Analytical")
plot_surface(x_mesh_, y_mesh_, z_noise, "Data with noise", show= True)
