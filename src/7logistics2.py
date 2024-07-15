import numpy as np
import matplotlib.pyplot as plt

nmax=500
def logistic(a,y0):
    y=[y0]        # yは1成分リストで、y[0]=y0
    for n in range(1,nmax): # n=1からnmax-1まで
        y.append(a * y[-1] * (1 - y[-1]))
    return y

y0=0.8
for a in np.linspace(0.5, 4.0, 1000): # aは0.5から4まで1000分轄
    yn = logistic(a,y0)[-100:] #計算結果はリストであり、最後の100要素
    an = [a]*len(yn)  # ynと同じ要素数のリストで、各成分は全てa
    plt.plot(an,yn, "c.", markersize=1.7)
plt.xlabel("alpha")
plt.ylabel("yn (n=399-499)")
plt.show()
