"""
@author: Anna Shirkalina (primarily), Genevive Beauregard
"""
import numpy as np
import matplotlib.pyplot as plt


def e_x_squared(x: float) -> float:
    """
    Returns $e^{-(x^2)}$
    :param x: float that is in input to the function
    :return: the output of the function
    """
    return np.exp(-1 * (x ** 2))


def de_x_squared_dx(x: float) -> float:
    """
    Returns $-2x e^{-(x^2)}$
    :param x: float that is in input to the function
    :return: the output of the function
    """
    return np.exp(-1 * (x ** 2)) * (-2 * x)


def forward_derivative(x, h, func) -> float:
    """
    Return the forward derivative at func(x) with the step size h
    :param x: float
    :param h: float
    :param func: the function who's derivative you are taking
    :return: float
    """
    return (func(x + h) - func(x)) / h


def central_difference(x, h, func) -> float:
    """
    Return the central difference scheme derivative at func(x) with the step
    size h
    :param x: float
    :param h: float
    :param func: the function who's derivative you are taking
    :return: float
    """
    return (func(x + h / 2) - func(x - h / 2)) / h


# initialize x, and empty lists for storing results
x_0 = 0.5
numerical_derivatives_e_x_2 = []
error_forward_deriv = []
error_central_scheme = []
# for loop to calculate the error in derivatives for given h values
for n in range(-16, 1):
    # calculate the derivatives using both methods
    fwd_deriv = forward_derivative(x_0, 10 ** n, lambda x: e_x_squared(x))
    central_deriv = central_difference(x_0, 10 ** n, lambda x: e_x_squared(x))

    # save the derivatives to print later
    numerical_derivatives_e_x_2.append(fwd_deriv)

    # save the errors
    error_forward_deriv.append(np.abs(fwd_deriv - de_x_squared_dx(x_0)))
    error_central_scheme.append(np.abs(central_deriv - de_x_squared_dx(x_0)))

# make an h_values array for ploting purposes
h_values = np.divide(1, np.power(np.ones(17,) * 10, np.arange(16, -1, -1)))

# plot for Q1 c and d

#plt.rc('text', usetex=True)             # use LaTeX for text
plt.rc('font', family='serif')          # use serif font
plt.rcParams.update({'font.size': 14})  # increase font size

plt.figure(1, figsize=(10, 5))
plt.loglog(h_values, error_forward_deriv, label="Error in forward derivative")
plt.ylabel("Error in the derivative")
plt.xlabel("Size of h")
#plt.savefig("C:\Docs\workspace\PHY407\image_Q1.pdf")
plt.legend(loc='lower right')


plt.figure(2, figsize=(10, 5))
plt.loglog(h_values, error_forward_deriv, label="Error in forward derivative")
plt.loglog(h_values, error_central_scheme, label="Error in central scheme")
plt.ylabel("Error in the derivative")
plt.xlabel("Size of h")
#plt.savefig("C:\Docs\workspace\PHY407\image_Q1.pdf")
plt.legend(loc='lower right')
plt.show()
print('The true value of the derivative of $e^{x^2}$', de_x_squared_dx(x_0))
i=0
for n in h_values:
    print(f"The numerical value of the derivative with h={n}",
          numerical_derivatives_e_x_2[i])
    print(f"The error between the two values is", error_forward_deriv[i])
    i += 1
