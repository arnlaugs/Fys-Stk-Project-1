import sys
sys.path.append('../functions')
from functions import *
import numpy as np



n_x=1000
# Make data.
x = np.sort(np.random.uniform(0, 1, n_x))
y = np.sort(np.random.uniform(0, 1, n_x))

x, y = np.meshgrid(x,y)
z = FrankeFunction(x, y)

#Transform from matricies to vectors
x_1=np.ravel(x)
y_1=np.ravel(y)
z_1=np.ravel(z)
n=int(len(x_1))


print(K_fold(x,y,z,10,0.1, "OLS"))
print(Bootstrap(x_1,y_1,z_1,10,0.1,"Ridge"))

     

    



