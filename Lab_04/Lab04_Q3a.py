"""
Lab3 PHY407
@author: Genevieve Beauregard (primarily), Anna Shirkalina 
Q3a
"""
import numpy as np
import matplotlib.pyplot as plt

# Exercise 6.10

def f(x, c): 
    """Function on the left hand side of the equation. 
    INPUT: x, c as per equation. 
    OUTPUT: 1 - e^{-cx}
    """
    return 1 - np.exp(-c*x)


def error(x, x_prime, c): 
    """Returns for relaxation error for a guess x_prime as per equation 6.83 in the
    textbook
    Input: x_prime the estimate we want the error on, x the previous estimate
    Output: error for x_prime
    """
    
    #derivative analytically using the previous value
    f_prime = c * np.exp(-c*x)
    
    return (x - x_prime)/(1 - 1/f_prime)


def solve_fx(accuracy, a, c): 
    """Solves x = f(x, c) for an initial guess x = a, with the x step of dx
    using relaxation. Based on lecture notes
    INPUT: error, dx the precision, a, c
    OUTPUT: Solution of x = f(x, c) 
    """

    # based on lecture notes replace accuracy with error     

    a_list = [a] # list of test values

    
    x = a 
    x_prime = f(x,c)

    while abs(error(x, x_prime, c)) > accuracy:
        #update x, x_prime
        x = x_prime
        x_prime = f(x_prime, c)
        a_list.append(f(a_list[-1], c)) # 

    return a_list[-1]




accuracy = 1e-6


a = 0.5



# Set c parameter list
dc = 0.01 # Set c step  
c_start = 0.01 # start of c values we test
c_stop = 3 #stop
c = np.arange(c_start, c_stop, dc)

# initialize solutions for x for every c 
x_sols = np.zeros(len(c))

# populate array
for i in range(len(c)):
    x_sols[i] = solve_fx(accuracy, a, c[i])
    

# configuring plots, rica/erik taught me this snippet to format fonts
plt.rc('text', usetex=True)             # use LaTeX for text
plt.rc('font', family='serif')          # use serif font
plt.rcParams.update({'font.size': 15})  # set font size 
plt.plot(c, x_sols)
plt.title("Plot of c versus solutions x, to an accuracy of $10^{-6}$")
plt.xlabel('c')
plt.ylabel('x')
plt.savefig('Q3a.pdf')






