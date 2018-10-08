import sys
sys.path.append('../functions')
from functions import *
import numpy as np
from random import random, seed
np.random.seed(4155)

n_x=1000
m=5

x = np.sort(np.random.uniform(0, 1, n_x))
y = np.sort(np.random.uniform(0, 1, n_x))

x, y = np.meshgrid(x,y)
z = FrankeFunction(x, y)

#Transform from matricies to vectors
x_1=np.ravel(x)
y_1=np.ravel(y)
n=int(len(x_1))
z_1=np.ravel(z)+ np.random.random(n) * 1



X= create_X(x_1,y_1,n=m)

Models=[OLS(),Lasso(0.01),Ridge(0.1)]
Titles=["OLS",r"Lasso $\alpha =0.001$", "Ridge $\alpha =0.1$"]

#Calculate and plot k-fold
model=Models[0]
model.fit(X,z_1)
z_predict=model.predict(X)
k_values=[1,5,10]
MSE_list=np.zeros(len(k_values))
R2_list=np.zeros(len(k_values))
Bias_list=np.zeros(len(k_values))
Variance_list=np.zeros(len(k_values))

MSE_list[0]=MSE(z_1,z_predict)
R2_list[0]=R2_Score(z_1,z_predict)
Variance_list[0]=variance(z_predict)
Bias_list[0]=bias(z_1,z_predict)

i=1
for k in k_values:
    if k!=1:
        MSE_list[i],R2_list[i],Bias_list[i],Variance_list[i]=K_fold(x_1,y_1,z_1,k,0.1, model,m=m)

        i+=1
print("OLS, 0.1 ", "MSE= ", MSE_list, "R2= ", R2_list)
