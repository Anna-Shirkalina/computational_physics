"""
Lab 2 Functions
Authors: Genevieve Beauregard, Anna Shirlkalina

"""
import numpy as np
import scipy.optimize as spop
import matplotlib.pyplot as plt
from scipy import special
from time import time

# Our integration functions

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



# Q2A Functions ------------------------------------------------------

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

        
        
def ordermag(number):
    """Q2a. This returns the order of magnitude of a float"""
    
    return np.math.floor(np.math.log(number, 10)) 


def dawson_timer(x: float, N: int,  samplesize: int, func):
    """Q2a. Times dawson integration function, func, for N slices and x times 
    and collects times in an array and returns average"""
    times = np.zeros(samplesize)
    for i in range(samplesize):
        start = time()
        func(x, N)
        end = time()
        times[i] = end - start
        
    return np.average(times) 

def error_Dtrap(x, N2): 
    """Returns error for Dawson trapezoidal integration method for N2 slices,
    with function input x. Refer to equation 5.28"""
    
    N1 = N2//2 # finds N1 for the lower slice integral
    I1 = D_trap(x, N1) # fewer slice integral
    I2 = D_trap(x, N2) # double slice integral
    
    epsilon = (1/3)* abs(I2 - I1)
    
    return epsilon

def error_Dsimp(x, N2): 
    """Returns error for Dawson Simpson's integration method for N2 slices,
    with function input x. Refer to equation 5.20"""
    
    N1 = N2//2 # finds N1 for the lower slice integral
    I1 = D_simp(x, N1) # fewer slice integral
    I2 = D_simp(x, N2) # double slice integral
    
    epsilon = abs((1/15)* (I2 -I1))
    
    return epsilon


#Q2B Functions ------------------------------------------------------

def J_underint(m: int, x:float, theta: float):
    """Returns function under the integral for a bessel function, with m and x 
    and theta input as per formula."""
    
    return np.math.cos(m*theta - x*np.math.sin(theta))

    

def J(m: int, x:float): 
    """Calculates bessel function for given m and x using simpson's rule for 
    N = 1000 slices.
    """

    # No of slices
    N = 1000
    
    # Define integral bounds
    a = 0
    b = np.pi

    # Integral with respect to theta
    I = simp(a, b , N, lambda theta: J_underint(m, x, theta))
    
    return (1/np.pi) * I


def obtain_Jarray(m, x_array):
    """Obtains the array of $J_m$ associated with an array input x_array"""
    
    J_m = np.zeros(len(x_array))
    for i in range(len(x_array)):
        J_m[i] = J(m, x_array[i])

    return J_m


def I(r:float, k:float): 
    """Returns intensity, given a k input and r input in SI units.
    Refer to equation in Question 5.4"""
    
    I = (J(1, k*r)/ (r*k))**2
    return I



        




    





    




