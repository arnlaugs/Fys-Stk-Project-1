# Importing functions from folder with common functions for project 1
import sys
sys.path.append('../functions')
from functions import FrankeFunction, MSE, R2_Score, create_X, plot_surface, calc_beta
import numpy as np




class Ridge():
    def __init__(self, lmbda=1.0):
        self.lmbda = lmbda


    def fit(self, X, y):
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
        beta : array_like, shape=[?]
            Regression parameters.
        """
        if len(y.shape) > 1:
            y = np.ravel(y)

        I = np.identity(X.shape[1])
        self.beta = np.linalg.inv(X.T.dot(X) + self.lmbda * I).dot(X.T).dot(y)

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




# Make data.
N = 1000
x = np.sort(np.random.uniform(0, 1, N))
y = np.sort(np.random.uniform(0, 1, N))
x_mesh_, y_mesh_ = np.meshgrid(x,y)

# Add noise
z_noise = FrankeFunction(x_mesh_, y_mesh_) + np.random.normal(scale = 1, size = (N,N))

# Regression
X = create_X(x_mesh_, y_mesh_)


R = Ridge()
print(R.fit(X, z_noise))


# Create best-fit matrix for plotting
x_r = np.linspace(0,1,N)
y_r = np.linspace(0,1,N)
x_mesh, y_mesh = np.meshgrid(x,y)
X_r = create_X(x_mesh, y_mesh)

z_reg = (R.predict(X_r)).reshape((N,N))
plot_surface(x_mesh, y_mesh, z_reg, "Ridge regression")


from sklearn.linear_model import Ridge
model = Ridge(alpha = 1, tol=0.1, fit_intercept=False).fit(X, np.ravel(z_noise))
print (model.coef_)
