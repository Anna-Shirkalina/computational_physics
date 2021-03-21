
"""
Lab4 PHY407
@author: Genevive Beauregard, Anna Shirkalina
Functions
"""

import numpy as np
from gaussxw import gaussxw


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



