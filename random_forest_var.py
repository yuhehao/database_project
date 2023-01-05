
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sklearn
import math
import pickle
from scipy import sparse
Z=[]
with open ("test", "rb") as f:
    a = f.read().splitlines()
# print(a)
for i in range(len(a)):
    if i % 2 == 1:
        a[i] = a[i].decode("utf-8").strip("b'").strip("\r\n'").strip("Random Forest Test Error: ")
        print(a[i])
        Z.append(float(a[i]))

est_list = [i for i in range(1,201,10)]
m_list = [i for i in range(5,101,4)]
X=[]
Y=[]
add = 0
for i in est_list:
    for j in m_list:
        if abs(add - len(Z)) < 0.01:
            break
        X .append(i)
        Y.append(j)
        add+=1
print(len(X))
fig = plt.figure()
ax1 = plt.axes(projection = '3d')

ax1.scatter3D(X,Y,Z,c=Z,cmap='rainbow')
ax1.plot3D(X,Y,Z,'gray')


plt.show()

