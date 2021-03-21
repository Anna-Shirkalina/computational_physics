
"""
Lab3 PHY407
@author: Genevive Beauregard, Anna Shirkalina
Functions
"""

import numpy as np
import scipy.optimize as spop
import matplotlib.pyplot as plt
from scipy import special
from time import time
from gaussxw import gaussxw,gaussxwab

# LAB 2 Functions ------------------------------------------------------


def trap(a: float, b: float, N: int, func):
    """Numerically computes the integral using the trapezoidal method of an
    input function func of a single variable, from a to b with N slices.
    Based on Eqn 5.3 of Newman."""

    # Calculate h width
    h = (b - a)/N

    # Calculate integral using trap method based on Eqn 5.3 of the textbook
    # This bit of code was taken from the lecture 2 jupyter notebook

    # The end bits
    s = 0.5*func(a) + 0.5*func(b)

    for k in range(1,N):  # adding the interior bits
        s += func(a+k*h)

    I_trap = s * h

    return I_trap


def simp(a: float, b: float, N: int, func):
    """Numerically computes the integral using the simpson's method of an
    input function func of a single variable, from a to b with N slices.
    Based on Eqn 5.9 of Newman"""


    # Obtain width of slice
    h = (b - a)/N

    # end bits
    s = func(a) + func(b)

    # loop over the odd bits
    odd_sum = 0 # odd accumalator varaible
    for k in range(1, N, 2):
        odd_sum += func(a + k* h)

    # Now the even terms
    even_sum = 0
    for k in range(2, N, 2):
        even_sum += func(a + k*h)

    # Integral value
    I = (1/3)* h *(s + 4*odd_sum + 2*even_sum)

    return I




def D_f(t: float):
    """Q2a. Function under the integral in Dawson's function. Takes float input
    and returns $e^{t^2}$. Intended as input for D_trap and D_simp"""

    return np.exp(t**2)


def D_trap(x: float, N: int):
    """Q2a. Numerically computes Dawsons function using the trapezoidal method.
    x is the same as per the notation in the question. N is the number of
    slices. Based on equation 5.3."""

    D = np.exp(-x**2)* trap(0, x, N, D_f)

    return D


def D_simp(x:float, N: int):
    """Q2a Numerically computes Dawsons function using the simpsons method.
    x is the same as per the notation in the question. N is the number of
    slices. Based on equation 5.9"""

    return simp(0,x, N, D_f) * np.exp(-x**2)


# Lab4 FUNCTIONS ------------------------------------------------------

def gauss(a: float, b: float, N: int, f):
    """Numerically computes the integral using the gauss method of an
    input function func of a single variable, from a to b with N slices.
    Input:
    a, lower integration bound, b upper, N number of slices, f function under
    integral
    Output:
    Gauss integral of f from a to b using N slices."""
    # Based on lecture notes and page 170 of newman
    # call gausswx for xi, wi

    x, w = gaussxw(N)

    # map them to the required integration domain
    xp = 0.5*(b - a)*x + 0.5*(b+a)
    wp = 0.5*(b - a)*w

    # initialize integral to 0.
    I = 0.
    # loop over sample points to compute integral
    for k in range(N):
        I += wp[k]* f(xp[k])

    return I


def error_relaxation(x, x_prime, deriv, w):
    """Returns for relaxation error for a guess x_prime as per equation 6.83 in the
      textbook
      Input: x_prime the estimate we want the error on, x the previous estimate,
      deriv the derivative of the function being estimated
      Output: error for x_prime
      """

    return (x - x_prime) / (1 - 1 / ((1 + w) * deriv(x) - w))


def relaxation_estimator(a, accuracy, func, deriv, w=0):
    """Solves x = f(x, c) for an initial guess x = a, with the x step of dx
    using relaxation. Based on lecture notes
    INPUT: the inital guess a for the solution, the desired accuracy, func: the
    function that is being solved and it's derivative
    OUTPUT: Solution of x = f(x, c), the number of iterations needed to reach
    desired accuracy
    """

    # based on lecture notes replace accuracy with error

    a_list = [a]  # list of test values

    x = a
    x_prime = (1 + w) * func(x) - x * w

    while abs(error_relaxation(x, x_prime, deriv, w)) > accuracy:
        # update x, x_prime
        x = x_prime
        x_prime = (1 + w) * func(x) - x * w
        a_list.append(x_prime)  #

    return a_list[-1], len(a_list)
