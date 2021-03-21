from numpy.random import random
import numpy as np
import matplotlib.pyplot as plt

N = 5000  # number of time steps
L = 101  # number of positions on the square grid

x_path = np.zeros(N)
y_path = np.zeros(N)
x_init = (L - 1) // 2
y_init = (L - 1) // 2

x_path[0] = x_init
y_path[0] = y_init

for t in range(N - 1):
    r = random()
    if 0 <= r < 0.25:  # move + x
        x_path[t + 1] = x_path[t] + 1
        y_path[t + 1] = y_path[t]
    elif 0.25 <= r < 0.5:  # move - x
        x_path[t + 1] = x_path[t] - 1
        y_path[t + 1] = y_path[t]
    elif 0.5 <= r < 0.75:  # move + y
        x_path[t + 1] = x_path[t]
        y_path[t + 1] = y_path[t] + 1
    else:  # move - y
        x_path[t + 1] = x_path[t]
        y_path[t + 1] = y_path[t] - 1
    # do not let particle escape from box. It can't move past L
    if x_path[t + 1] > L:
        x_path[t + 1] = L
    if y_path[t + 1] > L:
        y_path[t + 1] = L
    if x_path[t + 1] < 0:
        x_path[t + 1] = 0
    if y_path[t + 1] < 0:
        y_path[t + 1] = 0


plt.figure(1)
plt.rc('font', family='serif')          # use serif font
plt.rcParams.update({'font.size': 13})  # set font size
plt.plot(x_path, y_path)
plt.xlabel("Position on x-axis unitless")
plt.ylabel("Position on y-axis unitless")
plt.title(f"2D Random walk with {N} time steps")
plt.xlim((0, L))
plt.ylim((0, L))
plt.show()
