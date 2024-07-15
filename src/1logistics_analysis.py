#　Euler法によるロジステック方程式の数値解
import numpy as np                    #数値計算用モジュールを読み込む
import matplotlib.pyplot as plt       #描画用のモジュールを読み込む
from diffrentialequations import Logistics

#他のパラメータを変えたい場合は、以下のように変更する
#tmax = 100
#dt = 0.1
#y0 = 10
#r = 0.1
#K = 100
#time,yexact,yeuler=Logistics(tmax=tmax,y0=y0,r=r,K=K,dt=dt).solve()

dt=0.1
time,yexact,yeuler=Logistics(dt=dt).solve()

dt2 = 1
time2,yexact2,yeuler2=Logistics(dt=dt2).solve()

dt3 = 5
time3,yexact3,yeuler3=Logistics(dt=dt3).solve()

plt.style.use('default')             #描画スタイルは通常の形

fig, ax = plt.subplots()
fig.suptitle("Logistic equation", fontsize = 16)
ax.plot(time ,yeuler, label = f"Euler Method dt={dt}")     #描画用データ作成(表示しないがここで図を作成)
ax.plot(time2,yeuler2, label = f"Euler Method dt={dt2}")     
ax.plot(time3,yeuler3, label = f"Euler Method dt={dt3}")     
ax.plot(time,yexact,linestyle="dotted", label = "Exact Solution")  

ax.legend()
ax.set_xlabel("days")             
ax.set_ylabel("y")

plt.show()                       #図を表示する
