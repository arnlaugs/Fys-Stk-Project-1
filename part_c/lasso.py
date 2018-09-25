# Importing functions from folder with common functions for project 1
import sys
sys.path.append('../functions')
from functions import FrankeFunction, MSE, R2_Score, create_X, plot_surface, calc_beta
import numpy as np
from sklearn.linear_model import Lasso


# Making meshgrid of datapoints
N = 1000
x = np.sort(np.random.uniform(0, 1, N))
y = np.sort(np.random.uniform(0, 1, N))
x_mesh_, y_mesh_ = np.meshgrid(x,y)
z = FrankeFunction(x_mesh_, y_mesh_)

# Add noise
z_noise = z + np.random.normal(scale = 1, size = (N,N))

# Perform regression
X = create_X(x_mesh_, y_mesh_)
model = Lasso(alpha=0.0001, fit_intercept=False)
model.fit(X, np.ravel(z_noise))

# Create best-fit matrix for plotting
x_r = np.linspace(0,1,N)
y_r = np.linspace(0,1,N)
x_mesh, y_mesh = np.meshgrid(x,y)
X_r = create_X(x_mesh, y_mesh)

z_reg = (model.predict(X_r)).reshape((N,N))
plot_surface(x_mesh, y_mesh, z_reg, "Lasso regression", show=True)

print("MSE: %.5f" %MSE(z, z_reg))
print("R2_Score: %.5f" %R2_Score(z, z_reg))
