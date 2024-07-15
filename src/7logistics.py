import numpy as np
import matplotlib.pyplot as plt

def logistic(a,y0):
    y=[y0]        # yは1成分リストで、y[0]=y0
    for n in range(1,21): # n=1から20まで
        y.append(a * y[-1] * (1 - y[-1]))
    return y
    
cn=1
for a in np.arange(0.5, 4.5, 0.5): # aは0.5から4.0まで0.5ステップ
    plt.subplot(4, 2, cn) #縦に4図、横に2図配置のcn番目の図
    for y0 in np.arange(0.1, 1.0, 0.1): # y0は0.1から0.9まで0.1ステップ
        yn=logistic(a,y0)
        plt.plot(yn,  ".-")
    plt.legend(["alpha="+ str(a)])
    cn += 1
plt.show()
