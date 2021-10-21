import numpy as np
from math import sin, cos
from numpy.linalg import norm
from matplotlib import pyplot as plt

Lmin = 0.924
Lmax = 1.15 * Lmin
h = 0.01
tf = 15

def calculate_ft(t):
    D = Lmin
    C = 0
    B = (3 * (Lmax - Lmin) / tf ** 2)
    A = (-2 / 3 * (B / tf))

    polynom = A * t ** 3 + B * t ** 2 + C * t + D

    return polynom

time_array = np.append(np.arange(0, tf, h), [tf])
row = time_array.shape
i = 0
polyarray = np.zeros(row)

for t in time_array:
    polyarray[i] = calculate_ft(t)
    i = i + 1

plt.plot(time_array, polyarray)
plt.show()