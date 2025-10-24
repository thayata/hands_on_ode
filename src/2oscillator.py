# Euler法　解析解の分かっている例と比較
import numpy as np                  #数値計算用モジュールを読み込む
import matplotlib.pyplot as plt     #描画用のモジュールを読み込む            

tmax=100    #初期値
dt=0.01     
y0=10
z0=2

y=[]        #配列（リスト）であることの宣言
z=[]
t=[]

def ExactSolution(t):
    return y0*np.cos(t)

def SinCos(y,z):
    return z, -y

def EulerOneStepForward(y,z,dt): 
    yy,zz=SinCos(y,z)
    return yy*dt + y, zz*dt+z

tprev=0
yprev=y0
zprev=z0
while tprev < tmax :
    y.append(yprev)
    z.append(zprev)
    t.append(tprev)
    yprev,zprev = EulerOneStepForward(yprev,zprev, dt)
    tprev += dt

plt.style.use('default')             #描画スタイルは通常の形

fig, ax = plt.subplots()
fig.suptitle("simple harmonic oscillation", fontsize = 16)
ax.plot(t,y, label = "Euler Method y")    
#ax.plot(t,z, label = "Euler Method z")  

yexact=[ ExactSolution(tt)  for tt in t]
ax.plot(t,yexact,linestyle="dotted", label = "Exact Solution")  

ax.legend()
ax.set_xlabel("days")             
ax.set_ylabel("y, z")            

plt.show()                       #図を表示する

