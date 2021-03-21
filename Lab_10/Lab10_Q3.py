"""
Lab10 PHY407
@author: Genevieve Beauregard(primarily), Anna Shirkalina 
Q3
"""
import numpy as np 
from numpy.random import normal
from random import random, uniform
import matplotlib.pyplot as plt


# Parameters for graphs ------------------------------------------------------

plt.rc('text', usetex=True)             # use LaTeX for text
plt.rc('font', family='serif')          # use serif font
plt.rcParams.update({'font.size': 17})  # set font size 

# Functions -------------------------------------------------------------------

# For normal distribution 

def f_a(x): 
    """
    Integrand to be integrated for part a 

    Parameters
    ----------
    x : float
        Function variable.

    Returns
    -------
    Float
        Value of integrand at a point x for part b.
    """
    
    return (x**(-0.5))/(1 + np.exp(x))


def f_b(x): 
    """
    Integrand to be integrated for part b

    Parameters
    ----------
    x : Float
        Function variable.

    Returns
    -------
    Float
        Value of integrand at a point x for part b.
    """
    return np.exp(-2 * abs(x - 5))


def p_a(x):
    """
    Probability distribution for part a

    Parameters
    ----------
    x : Float
        x-coordinate.

    Returns
    -------
    Float
        Probability of the x-coordinate.

    """
    return 1/(2 * np.sqrt(x))

def p_b(x): 
    """
    Normal distribution as per 2b

    Parameters
    ----------
    x : Float

    Returns
    -------
    Float
        Probablity of getting x value.
    """
    return (1 / np.sqrt(2 * np.pi))*np.exp((-(x - 5)**2)/2)


def generaterandom_a(): 
    """
    Gives out number between 0 and 1 according to probability distribution p_a. 

    Returns
    -------
    Float
        Random variable.

    """
    
    return random() **2 



def generaterandom_b(): 
    """
    Gives out number between 0 and 10 according to normal distribution given by 
    b.
    Returns
    -------
    Float
        Random variable.
    """
    mu = 5
    sigma =1
    return normal(mu, sigma) 



def ImportSampMontecarlo(f, p, N, generaterandom): 
    """
    Implements importance sampling for montecarlo, where f is the function to
    be integrated, p is the resultant probability distribution from
    the weighting function. 


    Parameters
    ----------
    f : Function
        Integrand.
    p : Function
        Integrand.
    N : Int
        Number of sample points.
    generaterandom : Function
        The function that generates random number in accordance to prob
        distribution p.

    Returns
    -------
    I : Float
        Integral value.

    """
    
    I = 0 
    for i in range(N):
        x = generaterandom()
        # add the weighted sum 
        I += f(x) / p(x)
    
    # divide by number of points
    I = (1/N)* I
    
    return I 



def MeanValueMontecarlo(f, a, b, N):
    """
    Mean value montecarlo integration.

    Parameters
    ----------
    f : Function
        Integrand to be integrated.
    a : Float
        Lower integration limit.
    b : Float
        Upper integration limit.
    N : Int
        Number of samples.

    Returns
    -------
    Float 
        Integral value.

    """
    
    I = 0 
    
    for i in range(N): 
        x = uniform(a,b)
        I += f(x)
    
    return ((b - a)/N) * I 


# QUESTION 3A #
# Setting parameters ----------------------------------------------------------

tries = 100 # number of times we run the integral 
N = 10000 # No of sample points


# Sampling method -------------------------------------------------------------

# Initiaize array to accumalate the integrals runs
I_samples = np.zeros(tries)
for i in range(tries): 
    I_samples[i] = ImportSampMontecarlo(f_a, p_a, N, generaterandom_a)
    
# Mean Values -----------------------------------------------------------------

I_meanvalues = np.zeros(tries)
for i in range(tries): 
    I_meanvalues[i] = MeanValueMontecarlo(f_a, 0, 1, N)

# Plot ------------------------------------------------------------------------

plt.figure()
plt.hist(I_samples, 10, range=[0.8, 0.88],color = 'black', edgecolor="white")
plt.ylim(0, 90)
plt.title("Q3a Importance Sampling Monte Carlo Values ")
plt.xlabel('Computed Integral Value')
plt.ylabel('Occurrences')
plt.tight_layout()
plt.savefig('Q3a_sampling.pdf')

plt.figure()
plt.ylim(0, 90)
plt.hist(I_meanvalues, 10, range=[0.8, 0.88],color = 'black',edgecolor="white")
plt.title("Q3a Mean Value Monte Carlo Values")
plt.xlabel('Computed Integral Value')
plt.ylabel('Occurrences')
plt.tight_layout()
plt.savefig('Q3a_meanvalue.pdf')
        
        
# QUESTION 3B #
# Setting parameters ----------------------------------------------------------

tries = 100
N = 10000

# Sampling method -------------------------------------------------------------

# Initiaize array to accumalate the integrals runs
I_samples = np.zeros(tries)
for i in range(tries): 
    I_samples[i] = ImportSampMontecarlo(f_b, p_b, N, generaterandom_b)

# Mean Value Method------------------------------------------------------------

I_meanvalues = np.zeros(tries)

for i in range(tries): 
    I_meanvalues[i] = MeanValueMontecarlo(f_b, 0, 10, N)

# Plot ------------------------------------------------------------------------

plt.figure()
plt.hist(I_samples, 10, range=[0.96, 1.04], color = 'black',edgecolor="white")
plt.ylim(0, 60)
plt.title("Q3b Importance Sampling Monte Carlo Values ")
plt.xlabel('Computed Integral Value')
plt.ylabel('Occurrences')
plt.tight_layout()
plt.savefig('Q3b_sampling.pdf')

plt.figure()
plt.ylim(0, 60)
plt.hist(I_meanvalues, 10,range=[0.96, 1.04],color = 'black', edgecolor="white")
plt.title("Q3b Mean Value Monte Carlo ")
plt.xlabel('Computed Integral Value')
plt.ylabel('Occurrenced')
plt.tight_layout()
plt.savefig('Q3b_meanvalue.pdf')
