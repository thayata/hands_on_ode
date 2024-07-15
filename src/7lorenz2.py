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
    xx = X[0]
    yy = X[1]
    zz = X[2]
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

#極大値を見つける
zpeakN= (np.diff(np.sign(np.diff(z))) < 0).nonzero()[0] + 1 # local max
zpeak = [z[n] for n in zpeakN]

fig = plt.figure()
ax = fig.add_subplot(1,2,1)
ax.plot(z)
ax.legend(["z"])
ax.plot(zpeakN, zpeak,".")

ax2 = fig.add_subplot(1,2,2,title="Return Map", xlabel="zpeak(n)", ylabel="zpeak(n+1)")
ax2.plot(zpeak[0:-1], zpeak[1:],".")
plt.show()
