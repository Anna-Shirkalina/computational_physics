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


def temp_function_ii(x, y):
    """Return the temperature function as given in equation 9"""
    return np.cos(x)+np.cos(np.sqrt(2)*x)+np.cos(np.sqrt(3)*x)+((y-1)**2)


N = 25
R = 0.02
Tmax = 10.0
Tmin = 1e-3
tau = 1e5

# Choose N city locations and calculate the initial distance
x, y = random.randint(0, 50), random.randint(-20, 20)
temperature = temp_function_ii(x, y)

# Main loop
t = 0
T = Tmax
num_iter = 0
values = [[x, y]]

while T > Tmin:
    num_iter += 1

    # Cooling
    t += 1
    T = Tmax*np.exp(-t/tau)

    # generate random numbers
    delta1, delta2 = guassian()
    x_old, y_old = x, y
    x, y = x + delta1, y + delta2

    # make sure x and y are within the desired boundary
    if not 0 < x < 50:
        x = x_old
    if not -20 < y < 20:
        y = y_old

    # Swap them and calculate the change in temperature
    oldTemperature = temperature
    temperature = temp_function_ii(x, y)

    deltaT = temperature - oldTemperature

    # If the move is rejected, swap them back again
    if random.random() > np.exp(-deltaT/T):
        x, y = x_old, y_old
        temperature = oldTemperature
    values.append([x, y])  # save the values

print(x, y)

print("Number of iterations", num_iter)

values = np.array(values)
font = {'family': 'DejaVu Sans', 'size': 11}  # adjust fonts
rc('font', **font)
plt.figure(1)
plt.plot(range(len(values[:, 0])), values[:, 0], '.', label="The x values over time")
plt.plot(range(len(values[:, 0])), values[:, 1], '.', label="The y values over time")
plt.title(r"""The values of x, y during simulated annealing 
where $f(x, y) = cos(x) + cos(\sqrt{2} x) + cos(\sqrt{3} x) + (y -1)^2$""")
plt.xlabel("Number of loop iterations")
plt.ylabel("Value of x, y")
plt.legend(loc="best")
plt.show()
