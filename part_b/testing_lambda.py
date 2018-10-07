# Importing functions from folder with common functions for project 1
import sys, time

sys.path.append('../functions')
from functions import *
from regression import Ridge, OLS, Lasso

import matplotlib.pyplot as plt
#from sklearn.linear_model import Lasso
import numpy as np




# Setting seed value
np.random.seed(79162)


# Making meshgrid of datapoints and compute Franke's function
N = 1000
x = np.sort(np.random.uniform(0, 1, N))
y = np.sort(np.random.uniform(0, 1, N))
x_mesh_, y_mesh_ = np.meshgrid(x,y)
z = FrankeFunction(x_mesh_, y_mesh_)

# Add noise
z_noise = z + np.random.normal(scale = 1, size = (N,N))


# Setting order of regression
order = 5

# Crate design matrix
X = create_X(x_mesh_, y_mesh_, n=order)

# Create best-fit matrix for plotting
x_r = np.linspace(0,1,N)
y_r = np.linspace(0,1,N)
x_mesh, y_mesh = np.meshgrid(x,y)
X_r = create_X(x_mesh, y_mesh, n=order)



num = 10                                  # Number of lambda values

# Creating array of lambda values
lmbdas = np.logspace(-4, 8, num)
R2_R = np.zeros(num)                      # Rigde R2-scores
R2_OLS = np.zeros(num)                    # Ordinary least squares R2-scores
R2_L = np.zeros(num)                      # Lasso R2-scores


for i in range(num):
    # Fitting OLS, Rigde and Lasso
    model_OLS = OLS(); model_OLS.fit(X, z_noise)
    model_R = Ridge(lmbda = lmbdas[i]); model_R.fit(X, z_noise)
    model_L = Lasso(alpha = lmbdas[i], fit_intercept=False); model_L.fit(X, np.ravel(z_noise))

    # Predicting Franke's function
    z_tilde_OLS = (model_OLS.predict(X_r)).reshape((N,N))
    z_tilde_R = (model_R.predict(X_r)).reshape((N,N))
    z_tilde_L = (model_L.predict(X_r)).reshape((N,N))

    # Saving the R2-score values
    R2_OLS[i] = R2_Score(z, z_tilde_OLS)
    R2_R[i] = R2_Score(z, z_tilde_R)
    R2_L[i] = R2_Score(z, z_tilde_L)

    update_progress("Regression:", i/float(num-1))


# Plotting the results
plt.semilogx(lmbdas, R2_OLS, label='OLS')
plt.semilogx(lmbdas, R2_R, label='Ridge')
plt.semilogx(lmbdas, R2_L, label='Lasso')
plt.grid()
plt.legend(fontsize=14)
plt.title(r'Varying $\lambda$ ' + 'for %dth order regression' %(order), size=14)
plt.xlabel(r'$\lambda$', size=13)
plt.ylabel('R2-score', size=13)
plt.tick_params(labelsize=13)
savefigure("lambdas")
plt.show()
