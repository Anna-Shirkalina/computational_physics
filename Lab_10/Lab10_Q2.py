"""
Lab10 PHY407
@author: Genevieve Beauregard(primarily), Anna Shirkalina 
Q2
"""


import numpy as np 
from random import random 
from scipy.special import gamma

# We reference the mcint.py as per pg 466 Newman
def f(r): 
    """
    Parameters
    ----------
    r : numpy Array of length n 
        Position vector
    
    Returns
    -------
    Int
        returns 1 if r is in the hypersphere inclusive of its border
        or 0 otherwise. 
    """
    f = 0 
    if np.linalg.norm(r) <= 1: 
        f = 1
    return f


def Volhypersphere(n, N):
    """
    Computes unit-hypersphere volume using montecarlo technique. 

    Parameters
    ----------
    n : int
         number of dimensions of the sphere.
    N : float (Ill convert it to int within the function)
        number of points used for montecarlo method
        

    Returns
    -------
    Float 
        Volume of a n-hypersphere of radius R.

    """
    N = int(N)
    
    count = 0 
    for i in range(N): 
        
        # Generate a random r 
        r = np.zeros(n)
        for j in range(n): 
            r[j] = random()
        
        count += f(r)
    # calculate integral
    I = (pow(2,n)/N)*count
    
    return I

# set to a million points
N = 1e6

print('Using N='+str(int(N))+' points for Monte Carlo integration')

# for circle
area = Volhypersphere(2, N)
actualarea = np.pi

# for sphere 
vol = Volhypersphere(10, N) 
actualvol = (np.pi**(10/2))/(gamma(10/2 + 1))

print('The approximated volume of a unit circle is '\
      +str(area))
print('The associated error is '+str(abs(area-actualarea)))

# expected error
expected_err = np.sqrt(actualarea*(4 - actualarea))/np.sqrt(N)
print('The expected error is {}'.format(expected_err))




print('')

print('The approximated volume of a 10-dim unit hypersphere is '\
      +str(vol))
print('The associated error is '+str(abs(actualvol-vol)))

expected_err = np.sqrt(actualvol*(2**10-actualvol))/np.sqrt(N)

print('The expected error is {}'.format(expected_err))






        
    
    


