# Importing functions from folder with common functions for project 1
import sys
sys.path.append('../functions')
from functions import *
from regression import OLS
import numpy as np

N = 1000
x = np.sort(np.random.uniform(0, 1, N))
y = np.sort(np.random.uniform(0, 1, N))
x_mesh_, y_mesh_ = np.meshgrid(x,y)
z = FrankeFunction(x_mesh_, y_mesh_)

# Add noise
z_noise = z + np.random.normal(scale = 1, size = (N,N))

# Plot Franke
fig, ax, surf = plot_surface(x_mesh_, y_mesh_, z, figsize = (2.64429, 1.98322))
fig.savefig("Franke.png")

# Create best-fit matrix for plotting
x_r = np.linspace(0,1,N)
y_r = np.linspace(0,1,N)
x_mesh, y_mesh = np.meshgrid(x,y)
model = OLS()

# Plot for higher order as well
for m in [2, 5, 10]:
	X = create_X(x_mesh_, y_mesh_, n=m)
	X_r = create_X(x_mesh, y_mesh, n=m)
	model.fit(X, z_noise)
	fig, ax, surf = plot_surface(x_mesh, y_mesh, (model.predict(X_r)).reshape((N,N)), figsize = (2.64429, 1.98322))
	fig.savefig("FrankeOrder%i.png" %m)