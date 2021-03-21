"""
Lab3 PHY407
@author: Genevieve Beauregard, Anna Shirkalina 
Q1b
"""
import numpy as np
import matplotlib.pyplot as plt
from time import time
from SolveLinear import GaussElim, PartialPivot
from numpy.random import rand



# No of N we want to test
N = 300
N_array = np.arange(1, N)


time_gausselim = np.zeros(len(N_array)) # Empty time array for gauss elim 
time_partialpivot = np.zeros(len(N_array)) # same for partial pivot solving
time_ludecomp = np.zeros(len(N_array)) # same LU 

err_gausselim = np.zeros(len(N_array))
err_partialpivot = np.zeros(len(N_array))
err_ludecomp = np.zeros(len(N_array))


def time_solve(A_in, v_in, function):
    """Times and returns error for solves A_in x = v system of equations for
    a solver function(A_in, V_in).
    Input: A_in, V_in
    Output: time to solve A_in x = v and error in tuple"""
    
    start = time()
    x = function(A_in, v_in)
    end = time()
    v_soln = np.dot(A_in, x)
    

    return (end - start), np.mean(abs(v_in-v_soln))




# loop over size
for i in range(len(N_array)): 
    v_in = rand(N_array[i])
    A_in = rand(N_array[i], N_array[i])
    
    #populate array
    time_gausselim[i], err_gausselim[i] = time_solve(A_in, v_in, GaussElim)
    time_partialpivot[i], err_partialpivot[i] = time_solve(A_in, v_in, PartialPivot)
    time_ludecomp[i], err_ludecomp[i] = time_solve(A_in, v_in, np.linalg.solve)
    



plt.rc('text', usetex=True)             # use LaTeX for text
plt.rc('font', family='serif')          # use serif font
plt.rcParams.update({'font.size': 15})  # set font size 


# Plot times
plt.figure(figsize=(10,5))
plt.semilogy(N_array, time_gausselim, label='GaussElim')
plt.semilogy(N_array, time_partialpivot, label='PartialPivot')
plt.semilogy(N_array, time_ludecomp, label='numpy.linalg.solve')
plt.legend()
plt.title('Times for solving Ax = v against size of A')
plt.xlabel('Size of matrix A')
plt.ylabel('Time (seconds)')
plt.savefig('Q1b_times.pdf')


# Plot errors
plt.figure(figsize=(10,5))
plt.semilogy(N_array, err_gausselim, label='GaussElim')
plt.semilogy(N_array, err_partialpivot, label='PartialPivot')
plt.semilogy(N_array, err_ludecomp, label='numpy.linalg.solve')
plt.legend()
plt.title('Error for solving Ax = v against size of A')
plt.xlabel('Size of matrix A')
plt.ylabel('Err')
plt.savefig('Q1b_err.pdf')