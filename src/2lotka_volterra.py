#  Euler法による Lotka-Volterra equations　の数値解
import numpy as np                    
import matplotlib.pyplot as plt       

tmax=100    #初期値
dt=0.01     
y0=10
z0=2
y=[]        #配列（リスト）であることの宣言
z=[]
t=[]
A, B, C, D = 0.3, 0.1, 0.3, 1.3

def LotVol(y,z,A,B,C,D):
    return A*y - B*y*z, C*y*z - D*z

def EulerOneStepForward(y,z,dt,A,B,C,D): 
    yy,zz=LotVol(y,z,A,B,C,D)
    return yy*dt + y, zz*dt+z

tprev=0
yprev=y0
zprev=z0
while tprev < tmax :
    y.append(yprev)
    z.append(zprev)
    t.append(tprev)
    yprev,zprev = EulerOneStepForward(yprev,zprev, dt,A,B,C,D)
    tprev +=  dt

plt.style.use('default')             #描画スタイルは通常の形

fig, ax = plt.subplots()
fig.suptitle("Lotka-Volterra equations", fontsize = 16)
ax.plot(t,y, label = "Euler Method y")    
ax.plot(t,z, label = "Euler Method z")  

ax.legend()
ax.set_xlabel("days")             
ax.set_ylabel("y, z")            

plt.show()                       #図を表示する
