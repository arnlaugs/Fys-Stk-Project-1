# Importing functions from folder with common functions for project 1
import sys
sys.path.append('../functions')
from functions import FrankeFunction, MSE, R2_Score, create_X, plot_surface, Bootstrap, K_fold
import numpy as np
import matplotlib.pyplot as plt


class Ridge():
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



    def predict(self, X):
        """
        Predicts the given parameters.

        Parameters
        -----------
        X : array_like, shape=[n_samples, n_features]
            Data

        Returns
        -------
        y_tilde : array_like
            The predicted values
        """

        y_tilde = X.dot(self.beta)
        return y_tilde




if __name__ == '__main__':
    np.random.seed(79162)
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

    sys.path.append('../part_a')
    from OLS import OLS

    lmbdas = np.linspace(0,15,100)
    MSEs = np.zeros(100)
    MSEs2 = np.zeros(100)
    for i in range(100):
        print(i)
        model2 = OLS(); model2.fit(X, z_noise)
        model = Ridge(lmbda = lmbdas[i]); model.fit(X, z_noise)
        z_tilde1 = (model.predict(X_r)).reshape((N,N))
        z_tilde2 = (model2.predict(X_r)).reshape((N,N))
        MSEs[i] = MSE(z, z_tilde1)
        MSEs2[i] = MSE(z, z_tilde2)

    plt.plot(lmbdas, MSEs, label='Ridge')
    plt.plot(lmbdas, MSEs2, label='OLS')
    plt.grid()
    plt.xlabel(r'$\lambda$')
    plt.legend()
    plt.ylabel('Mean square error')
    plt.show()

    #from sklearn.linear_model import Ridge
    #model = Ridge(alpha = 1, tol=0.1, fit_intercept=False).fit(X, np.ravel(z_noise))
    #print (model.coef_)
