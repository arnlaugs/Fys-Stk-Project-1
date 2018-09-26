# Importing functions from folder with common functions for project 1
import sys
sys.path.append('../functions')
from functions import *
import numpy as np



class OLS(REGRESSION):
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
            Regression parameters. Returns beta if ret=True.
        """
        if len(y.shape) > 1:
            y = np.ravel(y)

        self.beta = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(y)

        if ret:
            return self.beta

