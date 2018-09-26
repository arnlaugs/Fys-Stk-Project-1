import sys
import numpy as np
from sklearn.linear_model import Lasso
import numpy as np

"""
A file for all common functions used in project 1
 - Frankefunction for computing the FrankeFunction
 - MSE for computing the mean squared error
 - R2_Score for computing the R2 score
 - create_X for creating the design matrix
 - plot_surface for plotting surfaces z(x,y)
"""



def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4


def MSE(y, y_tilde):
	"""
	Function for computing mean squared error.
	Input is y: analytical solution, y_tilde: computed solution.
	"""
	return np.sum((y-y_tilde)**2)/y.size

def R2_Score(y, y_tilde):
	"""
	Function for computing the R2 score.
	Input is y: analytical solution, y_tilde: computed solution.
	"""

	return 1 - np.sum((y-y_tilde)**2)/np.sum((y-np.average(y))**2)


def create_X(x, y, n = 5):
	"""
	Function for creating a X-matrix with rows [1, x, y, x^2, xy, xy^2 , etc.]
	Input is x and y mesh or raveled mesh, keyword agruments n is the degree of the polinomial you want to fit.
	"""
	if len(x.shape) > 1:
		x = np.ravel(x)
		y = np.ravel(y)

	N = len(x)
	l = int((n+1)*(n+2)/2)		# Number of elements in beta
	X = np.zeros((N,l))

	for i in range(1,n+1):
		q = int((i)*(i+1)/2)
		for k in range(i+1):
			X[:,q+k] = x**(i-k) + y**k

	return X


def plot_surface(x, y, z, title, show = False, trans = False):
	"""
	Function to plot surfaces of z, given an x and y.
	Input: x, y, z (NxN matrices), and a title (string)
	"""

	from mpl_toolkits.mplot3d import Axes3D
	import matplotlib.pyplot as plt
	from matplotlib import cm
	from matplotlib.ticker import LinearLocator, FormatStrFormatter

	fig = plt.figure()
	ax = fig.gca(projection='3d')

	if trans:
		z = z.T
	# Plot the surface.of the best fit
	surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

	# Customize the z axis automatically
	z_min = np.min(z)
	z_min = z_min*1.01 if z_min < 0 else z_min*.99
	z_max = np.max(z)
	z_max = z_max*1.01 if z_max > 0 else z_max*.99

	ax.set_zlim(z_min, z_max)
	ax.zaxis.set_major_locator(LinearLocator(10))
	ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

	# Add a color bar which maps values to colors.
	fig.colorbar(surf, shrink=0.5, aspect=5)
	ax.set_title(title)

	if show:
		plt.show()

	return fig, ax ,surf


def train_test_data(x_,y_,z_,i):
	"""
	Takes in x,y and z arrays, and a array with random indesies iself.
	returns learning arrays for x, y and z with (N-len(i)) dimetions
	and test data with length (len(i))
	"""
	x_learn=np.delete(x_,i)
	y_learn=np.delete(y_,i)
	z_learn=np.delete(z_,i)
	x_test=np.take(x_,i)
	y_test=np.take(y_,i)
	z_test=np.take(z_,i)

	return x_learn,y_learn,z_learn,x_test,y_test,z_test


def calc_beta(X, z):
	"""
	Function for returning beta for ordinary least square regression
	Using pseudo inverse when the matrix is singular
	"""
	if len(z.shape) > 1:
		z = np.ravel(z)

	return np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(z)

def Bootstrap(x,y,z,k,alpha, method="OLS"):
    """Function to who calculate the average MSE and R2 using bootstrap.
    Takes in x,y and z varibles for a dataset, k number of times bootstraping,alpha and which method beta shall use. (OLS,Ridge or lasso)
    Returns average MSE and average R2"""

    if len(x.shape) > 1:
        x = np.ravel(x)
        y = np.ravel(y)
        z = np.ravel(z)

    n=len(x)
    i=np.arange(n)
    antall=int(n*0.1)
    MSE_=0
    R2_=0

    for t in range(k):
        x_,y_,z_,x_test,y_test,z_test=train_test_data(x,y,z,np.random.choice(n,antall,replace=False))
        X= create_X(x_,y_)
        X_test= create_X(x_test,y_test)
        if method=="OLS":
            sys.path.append('../part_a')
            from OLS import OLS
            model=OLS()
        elif method=="Ridge":
            sys.path.append('../part_b')
            from ridge import Ridge
            model = Ridge(lmbda=alpha)

        elif method=="Lasso":
            model = Lasso(alpha=alpha, fit_intercept=False)

        else:
            print("Wrong method, try either Lasso, Ridge or OLS")

        model.fit(X,z_)
        z_predict=model.predict(X_test)
        MSE_+=MSE(z_test,z_predict)
        R2_+=R2_Score(z_test,z_predict)

        """
        if MSE(z_test,zpredict) <MSE_:
            MSE_=MSE(z_test,zpredict)
            beta_MSE=beta
            l=t
        if R2_Score(z_test,zpredict)>R2_:
            R2_=R2_Score(z_test,zpredict)
            beta_R2=beta
            o=t
     print(o,R2_,l,MSE_)
     print(beta_MSE)
     print(beta_R2)
     """
    return (MSE_/k,R2_/k)

def K_fold(x,y,z,k,alpha,method="OLS"):
    """Function to who calculate the average MSE and R2 using k-fold.
    Takes in x,y and z varibles for a dataset, k number of folds, alpha and which method beta shall use. (OLS,Ridge or Lasso)
    Returns average MSE and average R2"""
    if len(x.shape) > 1:
        x = np.ravel(x)
        y = np.ravel(y)
        z = np.ravel(z)
    n=len(x)
    n_k=int(n/k)
    if n_k*k!=n:
        print("k needs to be a multiple of ", n)
    i=np.arange(n)
    np.random.shuffle(i)

    MSE_=0
    R2_=0
    for t in range(k):
        x_,y_,z_,x_test,y_test,z_test=train_test_data(x,y,z,i[t*n_k:(t+1)*n_k])
        X= create_X(x_,y_)
        X_test= create_X(x_test,y_test)
        if method=="OLS":
            sys.path.append('../part_a')
            from OLS import OLS
            model=OLS()
        elif method=="Ridge":
            sys.path.append('../part_b')
            from ridge import Ridge
            model = Ridge(lmbda=alpha)

        elif method=="Lasso":
            model = Lasso(alpha=alpha, fit_intercept=False)

        else:
            print("Wrong method, try either Lasso, Ridge or OLS")

        model.fit(X,z_)
        z_predict=model.predict(X_test)
        MSE_+=MSE(z_test,z_predict)
        R2_+=R2_Score(z_test,z_predict)
    return (MSE_/k,R2_/k)

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

