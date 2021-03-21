# SolveLinear.py
# Python module for PHY407
# Paul Kushner, 2015-09-26
# Modifications by Nicolas Grisouard, 2018-09-26
# This module contains useful routines for solving linear systems of equations.
# Based on gausselim.py from Newman
from numpy import empty
# The following will be useful for partial pivoting
# from numpy import empty, copy

#
import numpy as np

def GaussElim(A_in, v_in):
    """Implement Gaussian Elimination. This should be non-destructive for input
    arrays, so we will copy A and v to
    temporary variables
    IN:
    A_in, the matrix to pivot and triangularize
    v_in, the RHS vector
    OUT:
    x, the vector solution of A_in x = v_in """
    # copy A and v to temporary variables using copy command
    A = np.copy(A_in)
    v = np.copy(v_in)
    N = len(v)

    for m in range(N):
        # Divide by the diagonal element
        div = A[m, m]
        A[m, :] /= div
        v[m] /= div

        # Now subtract from the lower rows
        for i in range(m+1, N):
            mult = A[i, m]
            A[i, :] -= mult*A[m, :]
            v[i] -= mult*v[m]

    # Backsubstitution
    # create an array of the same type as the input array
    x = empty(N, dtype=v.dtype)
    for m in range(N-1, -1, -1):
        x[m] = v[m]
        for i in range(m+1, N):
            x[m] -= A[m, i]*x[i]
    return x


def PartialPivot(A_in, v_in):
    """ In this function, code the partial pivot (see Newman p. 222)
    IN:
    A_in, the matrix to pivot and triangularize
    v_in, the RHS vector
    OUT:
    x, the vector solution of A_in x = v_in """
    N = len(A_in)

    for i in range(N):
        
        # set max entry to the first i,i term
        max_entry = abs(A_in[i][i])
        
        # get the max row index
        max_rowindex = i
        
        # find max element to perform switch
        for j in range(i+1, N):
            if abs(A_in[j][i]) > max_entry:
                #set new max
                max_rowindex = j
                max_entry = abs(A_in[j][i]) 
        
        

        # Switch ith row with max for matrix and v
        A_in[i,:], A_in[max_rowindex,:] = np.copy(A_in[max_rowindex,:]), np.copy(A_in[i,:])
    
        v_in[i], v_in[max_rowindex] = np.copy(v_in[max_rowindex]), np.copy(v_in[i])
    
    return GaussElim(A_in, v_in)





# # Check step
                
# A_in = np.array([[3.0,4.0,-1.0, -1.0],
#         [ 1.0, -4.0,  1.0,  5.0],
#         [ 2.0,  1.0,  4.0,  1.0],
#         [ 2.0, -2.0,  1.0,  3.0]])

# v_in =[ 3.,  9., -4.,  7.]

# print(type(A_in))


# print(PartialPivot(A_in, v_in))

                

        
        
        

