import numpy as np
from sklearn.linear_model import Lasso as scikit_Lasso
from scipy.linalg import solve_triangular
from functions import *

class REGRESSION():
	"""
	Superclass for regression types
	"""

	def __init__(self):
		pass

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

	def store_beta(self, name):
		"""
		Saves computed beta-values

		Parameters
		-----------
		Name : string, name of file to store values to

		Returns
		-------

		"""

		np.save(name, self.beta)

	def load_beta(self, name):
		"""
		Loads stored beta-values

		Parameters
		-----------
		Name : string, name of file values are stored in

		Returns
		-------

		"""

		self.beta = np.load(name)




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
		"""
		try:
			Q, R = np.linalg.qr(X)

			c1 = Q.T.dot(y)

			self.beta = solve_triangular(R, c1)

		except np.linalg.linalg.LinAlgError:
		"""	
		self.beta = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(y)


		if ret:
			return self.beta



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


class Lasso(REGRESSION):
	def __init__(self, alpha=1e-10, fit_intercept=False):

		self.model = scikit_Lasso(alpha=alpha, fit_intercept=fit_intercept)

	def fit(self, X, y, ret=False):
		"""
		Fits the model using Lasso from scikit learn

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

		self.model.fit(X, y)

		self.beta = self.model.coef_

		if ret:
			return self.beta
