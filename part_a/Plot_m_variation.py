import sys
sys.path.append('../functions')
from functions import *
from regression import OLS
import sys

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

import numpy as np
from random import random, seed

from imageio import imread
np.random.seed(4155)

n_x=1800
m_values=[2,3,4,5,6,7,8,10,15,20]

x = np.sort(np.random.uniform(0, 1, n_x))
y = np.sort(np.random.uniform(0, 1, n_x))

x, y = np.meshgrid(x,y)

eval_terrain = False		# Set to true to evaluate terrain
if eval_terrain:
	terrain = imread('../part_d/n59_e010_1arc_v3.tif')
	z=terrain = terrain[:n_x,:n_x]
else:
	z = FrankeFunction(x, y)

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
betastds = []
betaavgs = []

for i in range(len(m_values)):
	MSE_list[i], R2_list[i], Bias_list[i], Variance_list[i], betastd, betaavg = K_fold(x_1, y_1, z_1, 5, 0.01, model,m=m_values[i])
	
	betaavgs.append(betaavg)
	betastds.append(betastd)

print("Confidence interval for m = %i" %m_values[0])
for i in range(int((m_values[0]+1)*(m_values[0]+2)/2)):
	print("Beta_%i: %.4f ± %.4f" %(i, betaavgs[0][i], 2*betastds[0][i]))



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

#savefigure('m_OLS', fig)
#plt.show()
