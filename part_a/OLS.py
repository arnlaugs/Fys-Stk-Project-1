# Importing functions from folder with common functions for project 1
import sys
sys.path.append('../functions')
from functions import *
from regression import OLS

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression as OLS_sklearn



# Making meshgrid of datapoints and compute Franke's function
N = 1000
x = np.sort(np.random.uniform(0, 1, N))
y = np.sort(np.random.uniform(0, 1, N))
x_mesh_, y_mesh_ = np.meshgrid(x,y)
z = FrankeFunction(x_mesh_, y_mesh_)

# Add noise
z_noise = z + np.random.normal(scale = 1, size = (N,N))

# Perform regression
X = create_X(x_mesh_, y_mesh_, n=2)
model = OLS()
beta = model.fit(X, z_noise, ret=True)

# Perform regression with Scikit learn
model2 = OLS_sklearn(fit_intercept = False)
model2.fit(X, np.ravel(z_noise))

# Print beta-values of the two models
print('============================')
print('Calculated beta-values:', beta)
print('Scikit-learn beta-values:', model2.coef_)

# Create best-fit matrix for plotting
x_r = np.linspace(0,1,N)
y_r = np.linspace(0,1,N)
x_mesh, y_mesh = np.meshgrid(x,y)
X_r = create_X(x_mesh, y_mesh, n=2)

# Predict
z_reg = (model.predict(X_r)).reshape((N,N))
plot_surface(x_mesh, y_mesh, z_reg, "OLS regression", show=True)
print('============================ \n')
print("MSE: %.5f" %MSE(z, z_reg))
print("R2_Score: %.5f" %R2_Score(z, z_reg))
