#　SIRモデル計算
import numpy as np                  
import matplotlib.pyplot as plt        

tmax=100        #初期値
dt=0.01     
N0 = 14000000    #　人口 1400万人
I0 = 10         #　初期感染者 10人
S0 = N0-10       #　未感染者　
R0 =0           #  初期免疫保持者　0
r = 1/14        #  14日程度で治癒する
b = 0.1*10/N0    #  p=0.1, m=10,  感染する確率

t = [ ]
S = [ ]
I = [ ]
R = [ ]
N = [ ]

def SIRmodel(S,I,R):
    return -b*S*I, b*S*I-r*I,r*I

def RungeKuttaOneStepForward(S,I,R, dt): 
    s1,i1,r1 =SIRmodel(S,I,R)
    s2,i2,r2 =SIRmodel(S+dt/2*s1, I+dt/2*i1, R+dt/2*r1)
    s3,i3,r3 =SIRmodel(S+dt/2*s2, I+dt/2*i2, R+dt/2*r2)
    s4,i4,r4 =SIRmodel(S+dt  *s3, I+dt  *i3, R + dt*r3)
    return (s1+2*s2+2*s3+s4)*dt/6 + S, (i1+2*i2+2*i3+i4)*dt/6 + I,  (r1+2*r2+2*r3+r4)*dt/6 + R

tprev=0
Sprev=S0
Iprev=I0
Rprev=R0

while tprev < tmax :
    S.append(Sprev)
    I.append(Iprev)
    R.append(Rprev)
    t.append(tprev)
    N.append(Sprev+Iprev+Rprev)
    Sprev,Iprev,Rprev = RungeKuttaOneStepForward(Sprev, Iprev,Rprev, dt)
    tprev += dt
    

plt.style.use('default')             #描画スタイルは通常の形

fig, ax = plt.subplots()
fig.suptitle("SIR model", fontsize = 16)
ax.plot(t,S, label = "S: susceptible")    
ax.plot(t,I, label = "I: infected")  
ax.plot(t,R, label = "R: recovered")    
ax.plot(t,N, label = "N: Total population")   

ax.legend()
ax.set_xlabel("days")             
ax.set_ylabel("S,I,R")            

plt.show()                       #図を表示する
