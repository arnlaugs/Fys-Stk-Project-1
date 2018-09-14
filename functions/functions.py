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


def plot_surface(x, y, z, title, show = False):
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

	# Plot the surface.of the best fit
	surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

	# Customize the z axis.
	ax.set_zlim(-0.10, 1.40)
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
