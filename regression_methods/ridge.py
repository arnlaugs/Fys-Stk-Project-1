# Importing functions from folder with common functions for project 1
import sys
sys.path.append('../functions')
from functions import FrankeFunction, MSE, R2_Score, create_X, plot_surface, calc_beta
import numpy as np


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
