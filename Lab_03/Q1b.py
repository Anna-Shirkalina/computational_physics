"""
Lab3 PHY407
@author: Genevive Beauregard(primarily), Anna Shirkalina 

Question 1b
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import special
from gaussxw import gaussxw,gaussxwab
import Functions as func
from numpy import pi, sin, arange
from pylab import plot, show, clf

def ubar(Ta: int, th: int): 
    """Calculates ubar for a given Ta and th scaler quantity as per equation 3.
    Input:
    T_a, average hourly temp in Celsius and snow surface age t_h, in hours
    Output:
    Mean wind speed ubar as per equation 3"""
    
    return 11.2 + 0.365*Ta + 0.00706*Ta**2 + 0.9* np.log(th)

def delta(Ta: int):
    """Calculates delta as per equation 4.
    Input:
    T_a, average hourly temp in Celsius
    Output:
    Standard deviation of windspeed, delta"""
    
    return 4.3 + 0.145 * Ta + 0.00196* Ta**2

def P_underint(Ta:int, th:int, u: int):
    """Function under the integral for P, equation 2. 
    Input: 
    u10, Ta, th scalar input
    Output:
    Function under the integral as per equation 2. """
    
    denominator = (ubar(Ta, th) - u)**2 
    numerator = 2 * delta(Ta)**2
    
    return np.exp(-denominator/numerator)

def Pfunc(u10: int, Ta: int, th: int, N: int):
    """Probability density function using gauss with N slices
    Input: 
    u10, Ta, th, and N number of slices
    Output: 
    Numerically computed P"""
    
    I = func.gauss(0, u10, N, lambda u: P_underint (Ta, th, u))
    
    fraction = 1/(np.sqrt(2 * np.pi)*delta(Ta)) 
    
    return fraction * I
    

# configuring plots, rica/erik taught me this snippet to format fonts
plt.rc('text', usetex=True)             # use LaTeX for text
plt.rc('font', family='serif')          # use serif font
plt.rcParams.update({'font.size': 15})  # set font size  
    

# Question 1 b setting parameters
Nb = 100
u10s = [6, 8, 10] # ms-1
ths = [24, 48, 72] # hours
    
# style parameters
colours = ('r', 'g', 'b')
lines = (':', '-', '--')

# create $T_a$ axis 
Tas = np.arange(-40, 30)

# empty Probability axis
P_arr = np.zeros(len(Tas))


plt.figure(figsize=(10,5))
# loop over th values with respective colour style
for (th, colour) in zip(ths, colours): 
    # loop over u10 values with respective line styles
    for (u10, line) in zip(u10s, lines):
        # create style string
        plot_str = colour + line
        
        # Create P array for given u10 and th
        for i in range(len(Tas)): 
            P_arr[i] = Pfunc(u10, Tas[i], th, Nb) 
        # plot Tas againts P_array
        plt.plot(Tas, P_arr,plot_str, label ='$u_{10}$ = '+ str(u10)+\
                 ' $ms^{-1}$, $t_h$ ='+str(th)+'hrs')
plt.title('$P(u_{10}, T_a, t_h)$ as per equation 2')
plt.xlabel('$T_a$ in $^o C$')
plt.ylabel('$P(T_a)$')
plt.legend()
plt.savefig('Q1b.pdf')
plt.show


          




