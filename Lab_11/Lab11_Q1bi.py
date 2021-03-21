import numpy as np
import matplotlib.pyplot as plt
import random as random
from matplotlib import rc

def guassian():
    """Return two random Guassian numbers,
    code copied from rutherford.py from Newman"""
    sigma = 1  # standard deviation of 1
    r = np.sqrt(-2 * (sigma ** 2) * np.log(1 - random.random()))
    theta = 2 * np.pi * random.random()
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y


def temp_function_i(x, y):
    """Return the temperature function as given in equation 9"""
    return x**2 - np.cos(4 * np.pi * x) + (y - 1)**2


N = 25
R = 0.02
Tmax = 10.0
Tmin = 1e-3
tau = 1e3

# Choose N city locations and calculate the initial distance
x, y = 2, 2
temperature = temp_function_i(x, y)

# Main loop
t = 0
T = Tmax
num_iter = 0

# initialize arrays
values = [[x, y]]

while T > Tmin:
    num_iter +=1

    # Cooling
    t += 1
    T = Tmax*np.exp(-t/tau)

    delta1, delta2 = guassian()
    x_old, y_old = x, y
    x, y = x + delta1, y + delta2

    # Swap them and calculate the change in temperature
    oldTemperature = temperature
    temperature = temp_function_i(x, y)

    deltaT = temperature - oldTemperature

    # If the move is rejected, swap them back again
    if random.random() > np.exp(-deltaT/T):
        x, y = x_old, y_old
        temperature = oldTemperature
    values.append([x, y]) # save the x and y values

print(x, y)
print("Number of iterations", num_iter)

values = np.array(values)
font = {'family': 'DejaVu Sans', 'size': 11}  # adjust fonts
rc('font', **font)
plt.figure(1)
plt.plot(range(len(values[:, 0])), values[:, 0], '.', label="The x values over time")
plt.plot(range(len(values[:, 0])), values[:, 1], '.', label="The y values over time")
plt.title(r"The values of x, y during simulated annealing for $f(x, y) = x^2 - cos(4 \pi x) + (y -1)^2$")
plt.xlabel("Number of loop iterations")
plt.ylabel("Value of x, y")
plt.legend(loc="best")
plt.show()
