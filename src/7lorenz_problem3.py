# 3元1階の常微分方程式なので、SIRと同じコードでもよいが、pythonらしく書いてみた。

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from diffrentialequations import Lorentz

x0=np.array([0,0,0])
t,X=Lorentz(r=14,x0=x0).solve()
x,y,z =X[:,0], X[:,1], X[:,2]

x02=x0+10**(-5)*np.array([0,1,0])
t,X=Lorentz(r=14,x0=x02).solve()
x2,y2,z2 =X[:,0], X[:,1], X[:,2]


#課題の(3)
#1logistics_analysis.py, 7lorentz.py, 7lorentz2.py, 7lorentz_analysis.pyのコードを参考にして、
# 以下にコードを追加して、初期条件を微小に変えた二つの場合のzが、時間とともに指数関数的に離れていく様子をプロットせよ。

plt.style.use('default')             #描画スタイルは通常の形


fig, ax = plt.subplots()
fig.suptitle("Lotka-Volterra equations", fontsize = 16)
#ax.plot(t,np.abs(y-y2), label = "Euler Method z")    

ax.plot(t,y, label = f"y0={x0[1]}")    
ax.plot(t,y2, label = f"y0={x02[1]}")    
#ax.set_xlim(0, .)
#ax.set_ylim(-5*10**(-1), 15**(1))

ax.legend()
ax.set_xlabel("t")             
ax.set_ylabel("z")            

plt.show() 