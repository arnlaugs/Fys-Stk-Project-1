import sys
sys.path.append('../functions')


from functions import *
sys.path.append('../part_b')
from ridge import Ridge


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
sys.path.append('../part_a')
from OLS import OLS
np.random.seed(4155)

n_x=1000
m=5

x = np.sort(np.random.uniform(0, 1, n_x))
y = np.sort(np.random.uniform(0, 1, n_x))
#y = np.arange(0, 1, 0.05)
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
"""
i=0
for model in Models:
    model.fit(X,z_1)
    z_predict=model.predict(X)
    fig, ax1, surf = plot_surface(x,y,z_predict.reshape(n_x,n_x),("Franke function fitted with "+ Titles[i]))
    i+=1
    plt.show()

"""
#Dette er for aa regne ut og plote K-fold
model=Models[0]
model.fit(X,z_1)
z_predict=model.predict(X)
#k_values=[1,2,3,4,5,6,8,9,10,12,15,16,18,20]#,25,40,50,100,200,500,800,1000]
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
    if k==1:
        pass
    #print(K_fold(x_1,y_1,z_1,k,0.1, "OLS"))
    else:
        MSE_list[i],R2_list[i],Bias_list[i],Variance_list[i]=K_fold(x_1,y_1,z_1,k,0.1, model,m=m)

        i+=1
print("OLS, 0.1 ", "MSE= ", MSE_list, "R2= ", R2_list)
"""
fig,ax1=plt.subplots()
ax1.plot(np.array(k_values),np.array(MSE_list),'x-',label="MSE")
#ax1.plot(np.array(k_values),Variance_list,'x-',label="Variance")
#ax1.plot(np.array(k_values),Bias_list,'x-',label="Bias")
ax1.set_xlabel('nuber of k-folds',fontsize=16)
ax1.set_ylabel('MSE',fontsize=16)
ax1.set_xticks(k_values)
ax2=ax1.twinx()
ax2.plot(np.array(k_values),R2_list,'x-',color='k',label="R2")
ax2.set_ylabel('R2 score',fontsize=16)
fig.legend(loc='upper right', bbox_to_anchor=(0.8, 0.9),fontsize=16)
plt.subplots_adjust(left=0.18, right=0.85)
plt.title("Varying K for OLS regression " r"$\alpha=0.1$", fontsize=16 )
#fig.tight_layout()
#plt.savefig('k_Lasso_franke.png')
plt.show()
"""
