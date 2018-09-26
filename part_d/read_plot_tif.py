# Importing functions from folder with common functions for project 1
import sys
sys.path.append('../functions')
sys.path.append('../part_a')      # Add folder with OLS
sys.path.append('../part_b')      # Add folder with Ridge
from functions import FrankeFunction, MSE, R2_Score, create_X, plot_surface
import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from OLS import OLS
from ridge import Ridge
from sklearn.linear_model import Lasso


# Load the terrain
terrain = imread('n59_e010_1arc_v3.tif')
print(np.shape(terrain))
N = 1800
terrain = terrain[:N,:N]

# Creates mesh of image pixels
x = np.linspace(0,1, np.shape(terrain)[0])
y = np.linspace(0,1, np.shape(terrain)[1])
x_mesh, y_mesh = np.meshgrid(x,y)

X = create_X(x_mesh, y_mesh)

R_ols = OLS();                  R_ols.fit(X, terrain)
R_r = Ridge(lmbda=1.0);         R_r.fit(X, terrain)
R_l = Lasso(alpha=0.1, fit_intercept=False); R_l.fit(X, np.ravel(terrain))



# Plotting surfaces and printing the MSE and R2-score
labels = ['Ordinary least squares', 'Ridge', 'Lasso']
i=0
for method in [R_ols, R_r, R_l]:
    z_reg = (method.predict(X)).reshape((N,N))
    plot_surface(x_mesh, y_mesh, z_reg.reshape((N,N)), labels[i], show=True, trans = True)

    print('============================')
    print(labels[i])
    print("MSE: %.5f" %MSE(terrain, z_reg))
    print("R2-score: %.5f \n" %R2_Score(terrain, z_reg))

    i+=1




# Plots surface plot
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(x_mesh, y_mesh, terrain.T, cmap=cm.viridis,
                   linewidth=0, antialiased=False)

# Shows image
plt.figure()
plt.title('Terrain over the Oslo area')
plt.imshow(terrain, cmap='gray')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
