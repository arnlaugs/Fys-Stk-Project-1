# Importing functions from folder with common functions for project 1
import sys
sys.path.append('../functions')
from functions import FrankeFunction, MSE, R2_Score, create_X, plot_surface, calc_beta
import numpy as np

# Importing the regression classes
from OLS import OLS
from ridge import Ridge
from sklearn.linear_model import Lasso




# Making meshgrid of datapoints
N = 100
x = np.sort(np.random.uniform(0, 1, N))
y = np.sort(np.random.uniform(0, 1, N))
x_mesh_, y_mesh_ = np.meshgrid(x,y)
z = FrankeFunction(x_mesh_, y_mesh_)

# Add noise
z_noise = z + np.random.normal(scale = 1, size = (N,N))

# Perform regression
X = create_X(x_mesh_, y_mesh_)
R_ols = OLS(); R_ols.fit(X, z_noise)
R_l = Lasso(alpha=0.1, fit_intercept=False); R_l.fit(X, np.ravel(z_noise))
R_r = Ridge(lmbda=1.0); R_r.fit(X, z_noise)


# Create best-fit matrix for plotting
x_r = np.linspace(0,1,N)
y_r = np.linspace(0,1,N)
x_mesh, y_mesh = np.meshgrid(x,y)
X_r = create_X(x_mesh, y_mesh)



# Plotting surfaces and printing the MSE and R2-score
labels = ['Ordinary least squares', 'Lasso', 'Ridge']
i=0
for method in [R_ols, R_l, R_r]:
    z_reg = (method.predict(X_r)).reshape((N,N))
    plot_surface(x_mesh, y_mesh, z_reg.reshape((N,N)), labels[i], show=True)

    print('============================')
    print(labels[i])
    print("MSE: %.5f" %MSE(z, z_reg))
    print("R2-score: %.5f \n" %R2_Score(z, z_reg))

    i+=1
