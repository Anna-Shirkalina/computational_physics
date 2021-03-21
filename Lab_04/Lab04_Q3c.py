"""
Lab3 PHY407
@author: Genevieve Beauregard, Anna Shirkalina (primarily)
Q3a
"""
import numpy as np
import matplotlib.pyplot as plt
import Functions as func

# exercise 6.13


def binary_search(x1, x2, epsilon, f, iteration=None):
    """Return the value of the zero of the function f,
    (or a close enough to the zero), and the number of iterations it took to
    calculate it
     Return -1, if there is no zero on the interval of x1 to x2
    """
    # set initial iteration call to 1 (the first time binary search is called)
    if iteration is None:
        iteration = 1
    # check if f(x1) happens to fall on a zero
    if f(x1) == 0:
        return x1, iteration
    # check to if f(x2) happens to fall on a zero
    if f(x2) == 0:
        return x2, iteration
    # if f(x1) and f(x2) have the same sign return -1, not valid inputs
    if not((f(x1) < 0 and f(x2) > 0) or (f(x1) > 0 and f(x2) < 0)):
        return -1
    # this means f(x1) and f(x2) have different signs
    else:
        # update x_prim and f_prime
        x_prime = 0.5 * (x1 + x2)
        f_prime = f(x_prime)
        # on the off change that you struck gold and f_prime = 0
        if f_prime == 0:
            return x_prime, iteration
        # update x1 if f_prime has a different sign than f(x1)
        if (f_prime > 0 and f(x1) > 0) or (f_prime < 0 and f(x1) < 0):
            x1 = x_prime
        # update x2 if f_prime has a different sign than f(x2)
        else:
            x2 = x_prime
        # calculate the error, if smaller than epsilon, return result
        if np.abs(x1 - x2) < epsilon:
            return 0.5 * (x1 + x2), iteration
        # else, add + 1 to the iteration and do another recursive call to the binary search
        else:
            iteration += 1
            return binary_search(x1, x2, epsilon, f, iteration)


def Newton_method(a, epsilon, func, deriv):
    """Solves func(x) = 0 using Newton's method within an error of epsilon.
    Input: initial guess a , error wanted epsilon, function func, derivative of 
    function deriv
    Output: x solution, number of iterations
    """

    x = a
    x_prime = x - func(x) / deriv(x)
    error = abs(x_prime - x)
    a_list = [a]  # list of test values
    while error > epsilon:
        # update x, x_prime
        x = x_prime
        x_prime = x - func(x) / deriv(x)
        a_list.append(x_prime)  #
        error = abs(x_prime - x)

    return a_list[-1], len(a_list)


def non_linear_equation(x):
    """The non-linear equation needed to find x, a variable in the Wien
    displacement constant
    f(x) = 0 = x + 5e^{-x} - 5
    """
    return 5 * np.exp(-x) + x - 5


def deriv_non_linear_equation(x):
    """The non-linear equation needed to find x, a variable in the Wien
    displacement constant
    f(x) = 0 = x + 5e^{-x} - 5
    """

    return -5 *np.exp(-x) + 1


def non_linear_equation_relaxation(x):
    """The non-linear equation needed to find x, a variable in the Wien
    displacement constant
    f(x) = x = 5 - 5e^{-x}
    """
    return 5 - 5 * np.exp(-x)


def deriv_non_linear_equation_relaxation(x):
    """The derivative of the non-linear equation for x, a variable in the Wien
        displacement constant
        f'(x) = 5e^{-x}
        """
    return 5 * np.exp(-x)


def b(x):
    """
    Return the Wein displacement constant, as a function of x.
    Where x = h c / (wave_lenght k_B T)
    """
    h = 6.62607004e-34 #J s
    c = 299792458  #m/s
    k_b = 1.38064852e-23 #J / K
    return h * c / (k_b * x)


# x = 0.5 gives a postive answer,
# x = -0.5 gives a negative answer so we can start there
# however x = 0 isn't a physical solution
# x = 6 is also a positive answer so we can try x1 = 4 and x2=6

wavelength = 502e-9  # in m

epsilon = 1e-6
# answer to question 6.13b
soln, iterations = binary_search(4, 6, epsilon, lambda x: non_linear_equation(x))
print("The solution to the non-linear equation using binary x=", soln)
print("We use", iterations,"iterations")
print("The Wien displacement constant is equal to b =", b(soln))
print("Therefore the temperature of the sun can be measured to be T = ", b(soln) / wavelength)


# using relaxation
soln, iterations = func.relaxation_estimator(4, epsilon,
                                             lambda x: non_linear_equation_relaxation(x),
                                             lambda y: deriv_non_linear_equation_relaxation(y))

print("")
print("The solution to the non-linear equation using relaxation x=", soln)
print("We use", iterations,"iterations")
print("The Wien displacement constant is equal to b =", b(soln))
print("Therefore the temperature of the sun can be measured to be T = ", b(soln) / wavelength)


# using Newton's

soln, iterations = Newton_method(4, epsilon, lambda x: non_linear_equation(x), lambda y: deriv_non_linear_equation(y))
print("")
print("The solution to the non-linear equation using Newton's x=", soln)
print("We use", iterations,"iterations")
print("The Wien displacement constant is equal to b =", b(soln))
print("Therefore the temperature of the sun can be measured to be T = ", b(soln) / wavelength)


# test performance over different starting values
x0 = np.arange(0.5, 5, 0.5)
binary_iter = np.zeros(x0.shape)
relaxation_iter = np.zeros(x0.shape)
newton_iter = np.zeros(x0.shape)
binary_soln = np.zeros(x0.shape)
relaxation_soln = np.zeros(x0.shape)
newton_soln = np.zeros(x0.shape)
x = 5 # the true value of x, or at least close enough
for i, a in enumerate(x0):
    soln, iter = binary_search(x - a,  x + a, epsilon, lambda x: non_linear_equation(x))
    binary_iter[i] = iter
    binary_soln[i]= soln

    soln, iter = func.relaxation_estimator(x - a, epsilon,lambda x: non_linear_equation_relaxation(x), lambda y: deriv_non_linear_equation_relaxation(y))
    relaxation_iter[i] = iter
    relaxation_soln[i] = soln

    soln, iter = Newton_method(x - a, epsilon, lambda x: non_linear_equation(x), lambda y: deriv_non_linear_equation(y))
    newton_iter[i] = iter
    newton_soln[i] = soln

plt.figure(1)
plt.rcParams.update({'font.size': 11})
plt.plot(x0, binary_iter, label="Binary Search")
plt.plot(x0, relaxation_iter, label="Relaxation method")
plt.plot(x0, newton_iter, label="Newton method")
plt.xlabel("Distance of initial guess from x=5")
plt.ylabel("Number of function iterations")
plt.legend(loc="best")


plt.figure(2)
plt.rcParams.update({'font.size': 11})
plt.plot(x0, binary_soln, label="Binary Search")
plt.plot(x0, relaxation_soln, label="Relaxation method")
plt.plot(x0, newton_soln, label="Newton method")
plt.xlabel("Distance of initial guess from x=5")
plt.ylabel("Solution of the method")
plt.legend(loc="best")
plt.show()

