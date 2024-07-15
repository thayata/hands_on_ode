# Runge=Kutta methodを用いたLotka-Volterra equations
import numpy as np                     
import matplotlib.pyplot as plt                   

tmax=100    #初期値
dt=0.001     
y0=10
z0=2
yE=[]        #Euler法
zE=[]
yR=[]        #RungeKutta法
zR=[]
t=[]
A, B, C, D = 0.3, 0.1, 0.3, 1.3

def LotVol(y,z):
    return A*y - B*y*z, C*y*z - D*z

def EulerOneStepForward(y,z,dt): 
    yy,zz=LotVol(y,z)
    return yy*dt + y, zz*dt+z

def RungeKuttaOneStepForward(y,z,dt): 
    k1,r1=LotVol(y,z)
    k2,r2=LotVol(y+dt/2*k1, z+dt/2*r1)
    k3,r3=LotVol(y+dt/2*k2, z+dt/2*r2)
    k4,r4=LotVol(y+dt*k3, z+dt*r3)
    return (k1+2*k2+2*k3+k4)*dt/6 + y, (r1+2*r2+2*r3+r4)*dt/6 +z

tprev=0
yprevE=y0
zprevE=z0
yprevR=y0
zprevR=z0

while tprev < tmax :
    yE.append(yprevE)
    zE.append(zprevE)
    yR.append(yprevR)
    zR.append(zprevR)
    t.append(tprev)
    yprevE,zprevE = EulerOneStepForward(yprevE,zprevE, dt)
    yprevR,zprevR = RungeKuttaOneStepForward(yprevR,zprevR, dt)
    tprev += dt

plt.style.use('default')             #描画スタイルは通常の形

fig, ax = plt.subplots()
fig.suptitle("Lotka-Volterra equations", fontsize = 16)
ax.plot(t,yE, label = "Euler Method y")    
ax.plot(t,zE, label = "Euler Method z")  
ax.plot(t,yR, label = "RungeKutta Method y",linestyle='dashed')    
ax.plot(t,zR, label = "RungeKutta Method z",linestyle='dashed')  

ax.legend()
ax.set_xlabel("days")             
ax.set_ylabel("y, z")            

plt.show()                       #図を表示する
