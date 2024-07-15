# 3元1階の常微分方程式なので、SIRと同じコードでもよいが、pythonらしく書いてみた。

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# const
p = 10
r = 28
b = 8/3
dt = 0.01
tmax = 50
x0,y0,z0=1, 1, 1  #初期条件
x = [ ]
y = [ ]
z = [ ]
t = [ ]

def OneStepForward(X,dt):  #Xはnumpyの配列(x,y,z)
    k1 = LorenzEquation(X)
    k2 = LorenzEquation(X + k1*dt/2)
    k3 = LorenzEquation(X + k2*dt/2)
    k4 = LorenzEquation(X + k3*dt)
    return  X + dt/6*(k1 + 2*k2 + 2*k3 + k4)

def LorenzEquation(X):    #Xはnumpyの配列(x,y,z)
    xx,yy,zz = X
    return np.array([-p*xx + p*yy, -xx*zz + r*xx - yy, xx*yy - b*zz])

tprev=0
xprev=x0
yprev=y0
zprev=z0
while tprev < tmax :
    x.append(xprev)
    y.append(yprev)
    z.append(zprev)
    t.append(tprev)
    [xprev,yprev,zprev] = OneStepForward([xprev,yprev,zprev], dt)
    tprev += dt

fig = plt.figure()
ax = fig.add_subplot(221, projection='3d')
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.plot(x,y,z)
ax2 = fig.add_subplot(2,2,2)
ax2.plot(x)
ax2.legend(["x"])
ax3 = fig.add_subplot(2,2,3)
ax3.plot(y)
ax3.legend(["y"])
ax4 = fig.add_subplot(2,2,4)
ax4.plot(z)
ax4.legend(["z"])
plt.show()
