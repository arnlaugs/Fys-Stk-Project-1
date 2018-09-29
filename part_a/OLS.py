# Importing functions from folder with common functions for project 1
import sys
sys.path.append('../functions')
from functions import *
import numpy as np
from scipy.linalg import solve_triangular



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


        #self.beta = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(y)
        
        Q, R = np.linalg.qr(X)

        c1 = Q.T.dot(y)

        self.beta = solve_triangular(R, c1)
        

        if ret:
            return self.beta
