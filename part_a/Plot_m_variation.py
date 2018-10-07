import sys
sys.path.append('../functions')
from functions import *
from regression import OLS

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

import numpy as np
from random import random, seed

from imageio import imread
np.random.seed(4155)

n_x=2000
m_values=[2,3,4,5,6,7,8,10,15,20]

x = np.sort(np.random.uniform(0, 1, n_x))
y = np.sort(np.random.uniform(0, 1, n_x))
#y = np.arange(0, 1, 0.05)
x, y = np.meshgrid(x,y)
z = FrankeFunction(x, y)

#terrain = imread('../part_d/n59_e010_1arc_v3.tif')
#z=terrain = terrain[:n_x,:n_x]
#Transform from matricies to vectors
x_1=np.ravel(x)
y_1=np.ravel(y)
z_1=np.ravel(z)
n=int(len(x_1))
model=OLS()

MSE_list=np.zeros(len(m_values))
R2_list=np.zeros(len(m_values))
Bias_list=np.zeros(len(m_values))
Variance_list=np.zeros(len(m_values))

i=0
for m in m_values:
	MSE_list[i],R2_list[i],Bias_list[i],Variance_list[i]=K_fold(x_1, y_1, z_1, 5, 0.01, model,m=m)

	i+=1


fig,ax1=plt.subplots()
line1, = ax1.plot(np.array(m_values),np.array(MSE_list),'x-',label="MSE")
line2, = ax1.plot(np.array(m_values),np.array(Variance_list),'x-',label="Variance")
line3, = ax1.plot(np.array(m_values),np.array(Bias_list),'x-',label="Bias")
ax1.set_xlabel('m',fontsize=12)
ax1.set_ylabel('MSE,Bias, Variance',fontsize=12)
ax1.set_xticks(m_values)
ax2=ax1.twinx()
line4, = ax2.plot(np.array(m_values),R2_list,'x-',color='k',label="R2")
ax2.set_ylabel('R2 score',fontsize=12)
plt.legend((line1, line2, line3, line4), ["MSE", "Variance", "Bias", "R2"], bbox_to_anchor=(0.75, 0.5),fontsize=12)
plt.subplots_adjust(left=0.18, right=0.85)
#plt.title("Varying m for OLS regression", fontsize=12 )

savefigure('m_OLS', fig)
plt.show()
