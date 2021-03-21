import numpy as np
from numpy import ones,copy,cos,tan,pi,linspace


def simpsons(a: float, b: float, f_array, h):
    """Numerically computes the integral using the simpson's method of an
    input function func of a single variable, from a to b with N slices.
    Based on Eqn 5.9 of Newman"""
    N = (f_array.shape[0])

    # # Obtain width of slice
    # h = (b - a) / N

    # end bits
    s = f_array[0] + f_array[-1]

    # loop over the odd bits
    odd_sum = 0  # odd accumalator varaible
    for k in range(1, N, 2):
        odd_sum += f_array[k]

    # Now the even terms
    even_sum = 0
    for k in range(2, N, 2):
        even_sum += f_array[k]

    # Integral value
    I = (1 / 3) * h * (s + 4 * odd_sum + 2 * even_sum)

    return I


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


def gaussxw(N):

    # Initial approximation to roots of the Legendre polynomial
    a = linspace(3,4*N-1,N)/(4*N+2)
    x = cos(pi*a+1/(8*N*N*tan(a)))

    # Find roots using Newton's method
    epsilon = 1e-15
    delta = 1.0
    while delta > epsilon:
        p0 = ones(N,float)
        p1 = copy(x)
        for k in range(1,N):
            p0,p1 = p1,((2*k+1)*x*p1-k*p0)/(k+1)
        dp = (N+1)*(p0-x*p1)/(1-x*x)
        dx = p1/dp
        x -= dx
        delta = max(abs(dx))

    # Calculate the weights
    w = 2*(N+1)*(N+1)/(N*N*(1-x*x)*dp*dp)

    return x,w
