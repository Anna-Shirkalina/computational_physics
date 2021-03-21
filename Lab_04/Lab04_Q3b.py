"""
Lab3 PHY407
@author: Genevieve Beauregard, Anna Shirkalina (primarily)
Q3a
"""
import numpy as np
import Lab4_Functions as func
import matplotlib.pyplot as plt
#from tabulate import tabulate # environment to print out values in a latex table

# exercise 6.11


def nonlinear_equation(x, c):
    """
    Return f(x) = 1 - exp(-c * x)
    :param x: the independent variable
    :param c: a parameter in the equation is being solved
    :return: f(x)
    """
    return 1 - np.exp(-c * x)


def deriv_nonlinear_equation(x, c):
    """
    Return the derivative of the non linear equation.
    f'(x) = c * exp(-c * x)
    :param x: the independent variable
    :param c: a parameter in the equation is being solved
    :return: f'(x)
    """
    return c * np.exp(-c * x)


# by graphing f(x)= 1- e^{-2x} - x, we can see that it has zeros at
# approximately 0, 0.8. So a = 0.7 is a good initial guess
accuracy = 1e-6

#find the answer
soln, iterations = func.relaxation_estimator(0.7, accuracy,
                                             lambda x: nonlinear_equation(x, 2),
                                             lambda y: deriv_nonlinear_equation
                                             (y, 2))
print((soln, iterations))



# set up loop with varying numbers of w to find optimal value
ws = np.arange(0, 1, 0.05)
solutions = np.zeros(ws.size)
iterations = np.zeros(ws.size)
for i, w_i in enumerate(ws):
    soln, iterat= func.relaxation_estimator(0.7, accuracy,
                                             lambda x: nonlinear_equation(x, 2),
                                             lambda y: deriv_nonlinear_equation
                                             (y, 2), w=w_i)
    solutions[i] = soln
    iterations[i] = iterat


# append the data together to print out a nice table
data = np.column_stack((ws, solutions, iterations))

#print(tabulate(data, tablefmt='latex_raw', floatfmt=".10f",
         #              numalign='left'))
