#　Euler法によるロジステック方程式の数値解
import numpy as np                    #数値計算用モジュールを読み込む

class Logistics:
    def __init__(
        self,
        tmax=100,
        y0=10,
        r=0.1,
        K=100,
        dt=0.1,
    ):
        self.tmax = tmax
        self.dt = dt
        self.y0 = y0
        self.r = r
        self.K = K
        self.T =int(tmax/dt)

    def logistics(self, y: float)->float:
        return self.r*(1-y/self.K)*y

    def EulerOneStepForward(self, y: float)->float:
        return self.logistics(y)*self.dt+y

    def solve(self):
        t=self.dt*np.arange(self.T+1)
        y =self.K/(1+ (self.K/self.y0-1)*np.exp(-self.r*t))
        y2=np.zeros(self.T+1)
        y2[0]=self.y0

        for i in range(self.T):
            y2[i+1]=self.EulerOneStepForward(y2[i])

        return t,y,y2
    


class Lorentz:
    def __init__(
        self,
        p = 10,
        r = 28,
        b = 8/3,
        dt = 0.01,
        tmax = 50,
        x0=np.array([1,1,1]),
    ):
        self.p = p
        self.r = r
        self.b = b
        self.dt = dt
        self.tmax = tmax
        self.x0 = x0
        self.T =int(tmax/dt)

    def LorenzEquation(self,X):    #Xはnumpyの配列(x,y,z)
        xx,yy,zz = X
        return np.array([-self.p*xx + self.p*yy, -xx*zz + self.r*xx - yy, xx*yy - self.b*zz])

    def OneStepForward(self,X):  #Xはnumpyの配列(x,y,z)
        k1 = self.LorenzEquation(X)
        k2 = self.LorenzEquation(X + k1*self.dt/2)
        k3 = self.LorenzEquation(X + k2*self.dt/2)
        k4 = self.LorenzEquation(X + k3*self.dt)
        return  X + self.dt/6*(k1 + 2*k2 + 2*k3 + k4)

    def solve(self):
        t=self.dt*np.arange(self.T+1)
        x=np.zeros((self.T+1,3))
        x[0,:]=self.x0

        for i in range(self.T):
            x[i+1,:]=self.OneStepForward(x[i,:])

        return t,x

if __name__ == '__main__':
    import matplotlib.pyplot as plt       #描画用のモジュールを読み込む

    dt=0.1
    Logi=Logistics(dt=dt)
    time,yexact,yeuler=Logi.solve()

    plt.style.use('default')             #描画スタイルは通常の形

    fig, ax = plt.subplots()
    fig.suptitle("Logistic equation", fontsize = 16)
    ax.plot(time ,yeuler, label = rf"Euler Method dt={dt}")     #描画用データ作成(表示しないがここで図を作成)
    ax.plot(time,yexact,linestyle="dotted", label = "Exact Solution")  

    ax.legend()
    ax.set_xlabel("days")             
    ax.set_ylabel("y")

    plt.show()                       #図を表示する
