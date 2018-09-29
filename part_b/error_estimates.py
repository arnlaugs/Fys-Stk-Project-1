# Importing functions from folder with common functions for project 1
import sys
sys.path.append('../part_a')
sys.path.append('../functions')
from functions import *
import matplotlib.pyplot as plt
import numpy as np
from ridge import Ridge
from OLS import OLS


# Making meshgrid of datapoints and compute Franke's function
x = np.linspace(0, 4*np.pi, 100)
z = np.sin(x)
z_noise = z + np.random.normal(scale = 0.3, size = (100))

x_pred = np.linspace(0, 4*np.pi, 100)


Xs = np.c_[np.ones((np.size(x),1)), x, x**2, x**3, x**4, x**5, x**6, x**7, x**8, x**9,\
x**(10), x**(11), x**(12), x**(13), x**(14), x**(15), x**(16), x**(17), x**(18), x**(19), x**(20)]
X_preds =  np.c_[np.ones((np.size(x_pred),1)), x_pred, x_pred**2, x_pred**3, x_pred**4, x_pred**5, x_pred**6, x_pred**7, x_pred**8, x_pred**9,\
x_pred**(10), x_pred**(11), x_pred**(12), x_pred**(13), x_pred**(14), x_pred**(15), x_pred**(16),\
x_pred**(17), x_pred**(18), x_pred**(19), x_pred**(20)]


# Orders of regression
n_order = 20
bias_R = np.zeros(n_order)
variance_R = np.zeros(n_order)

for i in range(n_order):
     # Perform regression
     X = Xs[:,:i+1]
     model = OLS()
     model.fit(X, z_noise, ret=True)

     # Predict
     X_pred = X_preds[:,:i+1]
     z_reg = model.predict(X_pred)

     bias_R[i] = bias(z, z_reg)
     variance_R[i] = variance(z_reg)

     plt.plot(x_pred, z_reg, label='predicted')
     plt.scatter(x, z_noise, alpha=0.5, label='datapoints')
     plt.plot(x, z, label='exact')
     plt.legend()
     plt.title('Order = %d' %i)
     plt.show()


     update_progress("Regression:", i/float(n_order-1))

plt.plot(np.arange(n_order), bias_R,  label=r'Bias$^2$')
plt.plot(np.arange(n_order), variance_R, label='Variance')
plt.plot(np.arange(n_order), bias_R + variance_R,'--', label='Bias$^2$ + variance')
plt.legend()
plt.xlabel('Order of regression')
plt.grid()
plt.show()









"""
# Making meshgrid of datapoints and compute Franke's function
N = 50
x = np.sort(np.random.uniform(0, 1, N))
y = np.sort(np.random.uniform(0, 1, N))
x_mesh_, y_mesh_ = np.meshgrid(x,y)
z = FrankeFunction(x_mesh_, y_mesh_)

# Add noise
z_noise = z + np.random.normal(scale = 1, size = (N,N))

# Create best-fit matrix for plotting
x_r = np.linspace(0,1,N)
y_r = np.linspace(0,1,N)
x_mesh, y_mesh = np.meshgrid(x,y)

# Orders of regression
n_order = 30


bias_R = np.zeros(n_order)
variance_R = np.zeros(n_order)


for i in range(n_order):
    # Perform regression
    X = create_X(x_mesh_, y_mesh_, n=i)
    model = OLS()
    model.fit(X, z_noise, ret=True)

    # Predict
    X_r = create_X(x_mesh, y_mesh, n=i)
    z_reg = (model.predict(X_r)).reshape((N,N))

    #plot_surface(x_mesh, y_mesh, z_reg, "Ridge regression", show=True)

    bias_R[i] = bias(z, z_reg)
    variance_R[i] = variance(z_reg)

    update_progress("Regression:", i/float(n_order-1))



plt.plot(np.arange(n_order), bias_R,  label=r'Bias$^2$')
plt.plot(np.arange(n_order), variance_R, label='Variance')
plt.plot(np.arange(n_order), bias_R + variance_R,'--', label='Bias$^2$ + variance')
plt.legend()
plt.xlabel('Order of regression')
plt.grid()
plt.show()
"""
