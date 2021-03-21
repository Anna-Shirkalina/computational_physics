"""
Question 2a
Authors: Genevieve Beauregard (Primary), Anna Shirlkalina
"""
import numpy as np
import scipy.optimize as spop
import matplotlib.pyplot as plt
from scipy import special
import Lab2_functions as func
from time import time

#Question 2a
N = 8
x = 4



print("Dawsons actual=>", special.dawsn(4))
print("")
print("For N =", N,",")
print("Our trap Dawson =>", func.D_trap(x, N))
print("Our simp Dawson =>", func.D_simp(x, N))

print("")



# Uncomment the below to get the error of for N slices  

# N2b_trap= 2**(2)
# error = abs(special.dawsn(4) - func.D_trap(x, N2b_trap))
# print("For N =", N2b_trap)
# print("Our trap Dawson =>", func.D_trap(x, N2b_trap))
# print("Error =", error)



# N2b_simp= 2**10
# error = abs(special.dawsn(x) - func.D_trap(x, N2b_simp))
# print("For N =", N2b_simp)
# print("Our simp Dawson =>", func.D_simp(x, N2b_simp))
# print("Error =", error)
# print("Error magnitude", func.ordermag(error))


print("")


# Q2 aii while for finding N for error o(-9)

#We do this for Trapezoidal first

#set N = powers of 2
n = 3
N2b_trap = 2**n

#Calculate error for trap
error = abs(special.dawsn(x) - func.D_trap(x, N2b_trap))

#If order is larger than -9 this loop will run
while func.ordermag(error) > -9:
    N2b_trap = 2**n
    error = abs(special.dawsn(x) - func.D_trap(x, N2b_trap))
    n +=1
    
print("For the trapezoid, we require",N2b_trap,\
      "slices for an error of o(10^{-9})")


 
# Same thing but for simp
n = 3
N2b_simp = 2**n
error = abs(special.dawsn(x) - func.D_simp(x, N2b_simp))
while func.ordermag(error) > -9:
    N2b_simp = 2**n
    error = abs(special.dawsn(x) - func.D_simp(x, N2b_simp))
    n +=1
    
print("For the simpsons, we require",N2b_simp,\
      "slices for an error of o(10^{-9})")
    

print("") # Gap for readability



# 2a ii timing functions
# Setting sample size to average over
sample_size = 1000

print("The average time to obtain this error for trapezoidal function=>",\
      func.dawson_timer( x, N2b_trap , sample_size, func.D_trap),\
          " seconds, using N = ", N2b_trap)
    

print("The average time to obtain this error for simpson's function=>",\
      func.dawson_timer( x, N2b_simp , sample_size, func.D_simp), \
          " seconds, using N = ", N2b_simp)
    

# timer for scipy 

sums = 0 # accumulator variable for times

for i in range(sample_size):
    # start clock
    start= time()
    # call funtion
    special.dawsn(x)
    # end clock
    end = time()
    # add to sum
    sums += end-start
    
# take average
special_time = sums/sample_size



print("In contrast, the average time to obtain this error for",\
      "scipy.special.dawson=>", special_time,"seconds")
    
    
#Q2a iii error estimation


#Set N2
N2 = 64

print("")

print("The error for N = ", N2,"for the trapezoidal method is =>",\
      func.error_Dtrap(x, N2))
    
    
print("The error for N = ", N2,"for the simpson's method is =>",\
      func.error_Dsimp(x, N2))

