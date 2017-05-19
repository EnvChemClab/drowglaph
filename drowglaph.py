import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from scipy.optimize import curve_fit

kion = pd.read_csv("data.csv")
kion2 = pd.read_csv("data2.csv")


def fit_func(x, a, b):
    return a * x**2 + b

res = curve_fit(fit_func, Px, Py)
res2 = curve_fit(fit_func, Qx, Qy)

a = res[0][0]
b = res[0][1]

c = res2[0][0]
d = res2[0][1]

Px2 = []
for x in Px:
    Px2.append(a * x**2 + b)

Qx2 = []
for x in Qx:
    Qx2.append(c* x**2 + d)

Px = kion['A']
Py = kion['V']

Qx = kion2['A']
Qy = kion2['V']

plt.xlim([0, 350])
plt.title("Fig.1")
plt.ylabel("voltage(V)")
plt.xlabel("amperage(mA)")
#plt.plot(Px, Py)
plt.plot(Px, np.array(Px2),  label="Electrolysis")
#plt.plot(Qx, Qy)
plt.plot(Qx, np.array(Qx2), label="Fuel cell")
plt.legend()
plt.show()
plt.savefig("VA _fig.png")
