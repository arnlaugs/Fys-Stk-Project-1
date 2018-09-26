# Importing functions from folder with common functions for project 1
import sys
sys.path.append('../functions')
from functions import *
import matplotlib.pyplot as plt
import numpy as np


class Ridge(REGRESSION):
    def __init__(self, lmbda=1.0):
        self.lmbda = lmbda


    def fit(self, X, y, ret=False):
        """
        Fits the model.

        Parameters
        -----------
        X : array_like, shape=[n_samples, n_features]
            Training data
        y : array-like, shape=[n_samples] or [n_samples, n_targets]
            Target values

        Returns
        --------
        beta : array_like, shape=[n_features]
            Regression parameters.
        """
        if len(y.shape) > 1:
            y = np.ravel(y)

        I = np.identity(X.shape[1])
        self.beta = np.linalg.pinv(X.T.dot(X) + self.lmbda * I).dot(X.T).dot(y)

        if ret:
            return self.beta






if __name__ == '__main__':
    # Making meshgrid of datapoints and compute Franke's function
    N = 1000
    x = np.sort(np.random.uniform(0, 1, N))
    y = np.sort(np.random.uniform(0, 1, N))
    x_mesh_, y_mesh_ = np.meshgrid(x,y)
    z = FrankeFunction(x_mesh_, y_mesh_)

    # Add noise
    z_noise = z + np.random.normal(scale = 1, size = (N,N))

    # # Perform regression
    X = create_X(x_mesh_, y_mesh_)
    # model = Ridge(lmbda=1)
    # model.fit(X, z_noise)
    #
    # Create best-fit matrix for plotting
    x_r = np.linspace(0,1,N)
    y_r = np.linspace(0,1,N)
    x_mesh, y_mesh = np.meshgrid(x,y)
    X_r = create_X(x_mesh, y_mesh)
    #
    # z_reg = (model.predict(X_r)).reshape((N,N))
    # plot_surface(x_mesh, y_mesh, z_reg, "Ridge regression", show=True)
    #
    # print("MSE: %.5f" %MSE(z, z_reg))
    # print("R2_Score: %.5f" %R2_Score(z, z_reg))


    from sklearn.linear_model import Ridge
    lmbdas = np.linspace(1e-4,1e8,1000)
    MSEs = np.zeros(1000)
    for i in range(1000):
        print(i)
        model = Ridge(alpha = lmbdas[i], fit_intercept=False).fit(X, np.ravel(z_noise))
        z_tilde = (model.predict(X_r)).reshape((N,N))
        MSEs[i] = MSE(z, z_tilde)
        #MSEs[i] = Bootstrap(x_mesh_, y_mesh_, z, k=10, lmbda=lmbdas[i])[0]

    plt.semilogx(lmbdas, MSEs)
    plt.grid()
    plt.xlabel(r'$\lambda$')
    plt.ylabel('Mean square error')
    plt.show()

    #from sklearn.linear_model import Ridge
    #model = Ridge(alpha = 1, tol=0.1, fit_intercept=False).fit(X, np.ravel(z_noise))
    #print (model.coef_)
