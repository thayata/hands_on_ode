import numpy as np                    #数値計算用モジュールを読み込む
import matplotlib.pyplot as plt       #描画用のモジュールを読み込む

tmax=100    #初期値
dt=0.1      #変えて見よ　例えば　1
y0=0.01
r =0.1
K=1
y=[]        #配列（リスト）であることの宣言
t=[]

def Exactlogistics(t,y0,r,K):   #微分方程式の解析解(オイラー法との比較)
    return K/(1+ (K/y0-1)*np.exp(-r*t))

def logistics(y, r, K):
    return r*(1-y/K)*y

def EulerOneStepForward(y, dt):  
    return logistics(y,r,K)*dt+y

tprev=0
yprev=y0
while tprev < tmax :
    y.append(yprev)
    t.append(tprev)
    yprev = EulerOneStepForward(yprev, dt)
    tprev +=  dt

y2=[]        #配列（リスト）であることの宣言
t2=[]
dt2=1

tprev=0
yprev=y0
while tprev < tmax :
    y2.append(yprev)
    t2.append(tprev)
    yprev = EulerOneStepForward(yprev, dt2)
    tprev +=  dt2

plt.style.use('default')             #描画スタイルは通常の形

fig, ax = plt.subplots()
fig.suptitle("Logistic equation", fontsize = 16)
ax.plot(t,y, label = f"Euler Method dt={dt}")     #描画用データ作成(表示しないがここで図を作成)
ax.plot(t2,y2, label = f"Euler Method dt={dt2}")     #描画用データ作成(表示しないがここで図を作成)

yexact=[Exactlogistics(tt,y0,r,K) for tt in t]
ax.plot(t,yexact,linestyle="dotted", label = "Exact Solution")  

ax.legend()
ax.set_xlabel("days")             
ax.set_ylabel("y")            

plt.show()                       #図を表示する
