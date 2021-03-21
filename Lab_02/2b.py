"""
Question 2b
Authors: Genevieve Beauregard (Primary), Anna Shirlkalina
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import special
import Lab2_functions as func


# Q2b i plots

# Parameters for x_array 
x_upp = 20
x_lower = 0
dx = 0.1

# x_array
x_array = np.arange(x_lower, x_upp, dx)

#J_m arrays 
J_0 = func.obtain_Jarray(0, x_array)
J_1 = func.obtain_Jarray(1, x_array)
J_2 = func.obtain_Jarray(2, x_array)

#Error between our J_m and scipy 
difference_0 =  abs(special.jv(0,x_array) - J_0)
difference_1 =  abs(special.jv(1,x_array) - J_1)
difference_2 =  abs(special.jv(2,x_array) - J_2)


# configuring plots, rica/erik taught me this snippet to format fonts
plt.rc('text', usetex=True)             # use LaTeX for text
plt.rc('font', family='serif')          # use serif font
plt.rcParams.update({'font.size': 15})  # set font size 



#Our bessel functions of order 1, 2, 3
plt.figure(figsize=(10,5))
plt.plot(x_array, J_0, label="$J_0$")
plt.plot(x_array, J_1, label="$J_1$")
plt.plot(x_array, J_2, label="$J_2$")
plt.title("Bessel Functions, calculated using Simpson's Method " +\
          "with N = 1000 slices")
plt.ylabel("$J_m(x)$")
plt.xlabel("$x$")
plt.legend()
plt.savefig("Q2bi.pdf")


#Difference between our J_m's and scipy's
plt.figure(figsize=(10,5))
plt.plot(x_array, difference_0, label="$|J(0,x) - special.jv(0,x)|$")
plt.plot(x_array, difference_1, label="$|J(1,x) - special.jv(1,x)|$")
plt.plot(x_array, difference_2, label="$|J(2,x) - special.jv(2,x)|$")
plt.title("Difference of our numerical $J(m,x)$ and scipy.special.jv(m, x)") 
plt.legend()
plt.ylabel("$|J(m, x) - scipy.special.jv(m, x)|$")
plt.xlabel("$x$")
plt.savefig("Q2bii_difference.pdf")



# Ex 5.4 b, where we plot a light intensity function
wavelength = 500e-09
k =  2 * np.pi / wavelength

# Create coords

# The bounds of the grid
upperbound = 1.5e-6
lowerbound = -1.5e-6# in meters

#step size/resolution
step = 0.1e-1  * 10**(-6)# in meters

# y and x arrays
y = np.arange(lowerbound,upperbound,step)
x = np.arange(lowerbound,upperbound,step)

# Empty light intensity grid
I_array = np.empty([len(x),len(y)] , float)

# Loop as per Ex 3.3 in textbook to calculate I grid entries
for j in range(len(y)):
    for i in range(len(x)):
        # Calculate r at [x[i], y[j]] coord
        r = np.sqrt(x[i]**2 + y[j]**2)
        
        # If r is 0 (or close within machine error) we set it to 0.25, per hint
        if abs(r) < 1e-20 : 
            I_array[i, j] = (0.5)**2
            
            # print(i, j)  # this was a check step to make sure only one coord 
            # # is registered as r = 0
        else: 
            #Use calculate intensity as per equation
            I_array[i, j] = func.I(r, k)
        
      
        

        

    

plt.figure()
plt.imshow(I_array,cmap='hot',\
            extent=(lowerbound, upperbound, lowerbound, upperbound), vmax=0.01)
plt.colorbar(label='Intensity')
plt.title("Diffraction Pattern")
plt.xlabel("x in m")
plt.ylabel("y in m")
plt.savefig("2b_diffract.pdf")






    

    
    




    
    
    