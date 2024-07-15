# 3元1階の常微分方程式なので、SIRと同じコードでもよいが、pythonらしく書いてみた。

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from diffrentialequations import Lorentz

x0=np.array([1,1,1])
t,X=Lorentz(x0=x0).solve()
x,y,z =X[:,0], X[:,1], X[:,2]

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