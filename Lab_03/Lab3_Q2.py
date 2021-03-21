import numpy as np
import scipy.special as fun
import matplotlib.pyplot as plt
from gaussxw import gaussxw
#from tabulate import tabulate

def H(n, x):
    """Return the hermite polynomial for H_n (x)"""
    H_0 = 1
    H_1 = 2 * x
    if n == 0:
        return H_0
    if n == 1:
        return H_1
    # return the first two defined values of H_n
    else:
        polynomials = [H_0, H_1] #keep all of the values of H in a list
        for j in range(2, n + 1):
            # calculate the next value of the polynomial iteratively
            H_n = 2 * x * polynomials[j - 1] - 2 * (j - 1) * polynomials[j - 2]
            polynomials.append(H_n)
        return polynomials[-1]


def harmonic_oscillator(n, x):
    """Return the harmonic oscillator function, as defined in the physics
    background"""
    return (1 / (np.sqrt((2 ** n) * (fun.factorial(n)) * np.sqrt(np.pi)))) * \
           np.exp(x**2 / -2) * H(n, x)


def d_harmonic_oscillator_dx(n, x):
    """Returns the derivative of the quantum harmonic oscillator, as defined in
    the physics background"""

    return (1 / (np.sqrt((2 ** n) * (fun.factorial(n)) * np.sqrt(np.pi)))) * \
           np.exp((x**2) / -2) * (-x * H(n, x) + 2 * n * H(n - 1, x))


def x_mean_square_integrand(n, x):
    """Return the integrand of the quantum  position uncertainty of the nth
    level of a quantum oscillator, as defined in the physics background with the
    coordinate transformation described on page 190 of Newman
    """
    z = np.tan(x)
    return (z ** 2) * np.abs(harmonic_oscillator(n, z)) ** 2 / (np.cos(x) ** 2)


def p_mean_square_integrand(n, x):
    """Return the integrand of the quantum  momentum uncertainty of the nth
    level of a quantum oscillator, as defined in the physics background with the
    coordinate transformation described on page 190 of Newman
    """
    z = np.tan(x)
    return np.abs(d_harmonic_oscillator_dx(n, z)) ** 2 / (np.cos(x) ** 2)


def energy_oscillator(x_ms, p_ms):
    """Return the total energy of the oscillator, as defined in the physics
    background"""
    return 0.5 * (x_ms + p_ms)


# calculate the sample points and weights

def gauss_weights(N, a, b):
    """Return the sample points, sample weights for guassian intergration,
    code taken from example 5.2 in Newman"""
    x, w = gaussxw(N)
    xp = 0.5 * (b - a) * x + 0.5 * (b + a)
    wp = 0.5 * (b - a) * w
    return xp, wp


def guassian_integration(xp, wp, func):
    """
    Return the value of the integation using Guassian quadrature,
    code taken from example 5.2 in Newman
    :param xp: the sample points
    :param wp: the sample weights
    :param func: the function being intergrated
    :return: float
    """
    s = 0.0
    for i in range(len(xp)):
        s += wp[i] * func(xp[i])
    return s



# Plot for Question 2 a
plt.figure(1)
#initializa x array
x_array = np.linspace(-4, 4, 100)
#loop over values of n
for n in range(0, 4):
    #initializa values of the hermite array
    hermite_n = np.zeros(x_array.shape)
    # calculate the hermite function over the range of x
    for i, x in enumerate(x_array):
        hermite_n[i] = H(n, x)
    #plot the hermite function
    plt.plot(x_array, hermite_n, label=f"Hermite polynomial n = {n}")
plt.legend(loc="best")
plt.xlabel("x")
plt.ylabel("Hermite polynomial")


plt.figure(2)
#initializa x array
x_array = np.linspace(-4, 4, 1000)
#loop over values of n
for n in range(0, 4):
    #initializa values of the wave function array
    wave_function_n = np.zeros(x_array.shape)
    # calculate the wave function over the range of x
    for i, x in enumerate(x_array):
        wave_function_n[i] = harmonic_oscillator(n, x)
    #plot the hermite function
    plt.plot(x_array, wave_function_n, label=f"Harmonic Oscillator for n = {n}")
plt.legend(loc="best")
plt.xlabel("x")
plt.ylabel("Harmonic Oscillator")



# Plot for Question 2 b

# Plot the hermite function
plt.figure(3)
x_array = np.linspace(-10, 10, 100)
hermite_n = np.zeros(x_array.shape)
n_30 = 30
# calculate at each x value
for i, x in enumerate(x_array):
    hermite_n[i] = H(n_30, x)
plt.plot(x_array, hermite_n, label=f"Hermite polynomial n = {n_30}")
plt.legend(loc="best")
plt.xlabel("x")
plt.ylabel("Hermite polynomial")

# Plot the Wavefunction
plt.figure(4)
x_array = np.linspace(-10, 10, 1000)
wave_function_30 = np.zeros(x_array.shape)
n_30 = 30
# calculate at each x value
for i, x in enumerate(x_array):
    wave_function_30[i] = harmonic_oscillator(n_30, x)
plt.plot(x_array, wave_function_30, label=f"Harmonic Oscillator for n = {n_30}")
plt.legend(loc="best")
plt.xlabel("x")
plt.ylabel("Harmonic Oscillator")


# calculations for Q3 c

# Integration for calculation of <p> and <x>
N = 100
a = - np.pi / 2
b = np.pi / 2

# get guassian weights
xp, wp = gauss_weights(N, a, b)

# initialize the arrays
n_array = np.arange(0, 16)
x_ms_array = np.zeros(n_array.shape)
p_ms_array = np.zeros(n_array.shape)
energy_array = np.zeros(n_array.shape)

# for loop to do the integration
for i, n in enumerate(n_array):
    x_ms = guassian_integration(xp, wp, lambda y: x_mean_square_integrand(n, y))
    p_ms = guassian_integration(xp, wp, lambda y: p_mean_square_integrand(n, y))
    x_ms_array[i] = x_ms
    p_ms_array[i] = p_ms
    energy_array[i] = energy_oscillator(x_ms, p_ms)


# take the square root of the <x>^2 and <p>^2 values
x_rms_array = np.sqrt(x_ms_array)
p_rms_array = np.sqrt(p_ms_array)

# append the data together to print out a nice table
data = np.column_stack((n_array.astype(int), x_ms_array, p_ms_array, energy_array, x_rms_array, p_rms_array))
headers = [r'$n$', r'$\braket{x^2}$', r'$\braket{p^2}$', r'$E$',
               r'$\sqrt{\braket{x^2}}$', r'$\sqrt{\braket{p^2}}$']
np.savetxt("Data_Q2c_data.txt", data, delimiter=" & ", header="N & $<x>$ & $<p>$ & E & <x> & <y>")
#print(tabulate(data, headers=headers, tablefmt='latex_raw', floatfmt=".10f",
                       numalign='left'))


plt.show()
