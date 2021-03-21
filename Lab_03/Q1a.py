"""
Lab3 PHY407
@author: Genevive Beauregard(primarily), Anna Shirkalina 

Question 1a
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import special
from gaussxw import gaussxw,gaussxwab


#---- COPIED FROM LAB 2 ---------
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
#------------------------------------------------------------------------------

def D_gauss(x: float, N: int): 
    """Numerically computes Dawson's function using Gauss' method. 
    INPUT:
    x as per question notation. N is the number of slices.
    OUTPUT:
    Dawson's Function for N slices and x """
    
    #From lecture notes and pg 170 Newman
    
    # call gausswx for xi, wi
    xi, w = gaussxw(N)
    
    # map them to the required integration domain
    xi = 0.5*(x)*xi + 0.5*(x)
    w = 0.5*(x)*w
    
    # initialize integral to 0.
    I = 0.
    # loop over sample points to compute integral
    for k in range(N):
        I += w[k]*D_f(xi[k])
    
    return I * np.exp(-x**2)

def ObtainErr_dgauss(x: float, N: int):
    """Numerically computes the error in D_gauss using equation 1 of handout.
    INPUT: 
    x as per question notation. N is the number of slices.
    OUTPUT: 
    Error for D_gauss(x,N)"""
    
    return D_gauss(x, 2*N) - D_gauss(x, N)


# configuring plots, rica/erik taught me this snippet to format fonts
plt.rc('text', usetex=True)             # use LaTeX for text
plt.rc('font', family='serif')          # use serif font
plt.rcParams.update({'font.size': 15})  # set font size 

# Set x 
x = 4

# Initialize N-array in powers of 2 from 8 to 2048
n = 3 # power of 2
N = [] 

while 2**n <= 2048: 
    N.append(2**n)
    n += 1

# print actual Dawson
print("Dawsons actual=>", special.dawsn(x))

# Loop over N to find numerical integrals and print for all three methods
for i in range(len(N)):
    print("")
    print("For N =", N[i],", x =", x)
    print("Our trap Dawson =>", D_trap(x, N[i]))
    print("Our simp Dawson =>", D_simp(x, N[i]))
    print("Our gauss Dawson =>",D_simp(x, N[i]))
    

    
# Initialise relative error array for three different methods of length N 
relerr_trap = np.zeros(len(N))
relerr_simp = np.zeros(len(N))
relerr_gauss = np.zeros(len(N))
err_gauss = np.zeros(len(N))




# loop over N and calculate error for each method and input into err array
for i in range(len(N)): 
    relerr_trap[i] = abs(special.dawsn(x) - D_trap(x, N[i]))
    relerr_simp[i] = abs(special.dawsn(x) - D_simp(x, N[i]))
    relerr_gauss[i] = abs(special.dawsn(x) - D_gauss(x, N[i]))
    err_gauss[i] = ObtainErr_dgauss(x, N[i])
    
plt.figure()
plt.title('Error as a function of N, N sampled from powers of 2')
plt.scatter(N, relerr_trap, label='Relative Error Trap',marker='x')
plt.scatter(N, relerr_simp, label='Relative Error Simp', marker='x')
plt.scatter(N, relerr_gauss, label='Relative Error Gauss', marker='x')
plt.scatter(N, err_gauss, label='Eqn 1 Error Gauss', marker='x')
plt.loglog()
#plt.legend(loc='upper right')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xlabel('N, number of slices')
plt.ylabel('Error')
plt.savefig('Q1ai_errs')
plt.show()







        
    