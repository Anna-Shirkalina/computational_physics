#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lab3 PHY407
@author: Genevive Beauregard(primarily), Anna Shirkalina 

Question 3
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import special
from gaussxw import gaussxw,gaussxwab
import Functions as func
import struct




def dwdx_central(w, h, i, j):
    """calculates dw/dy[i,j] using central difference and 2h interval""" 
    return (w[i +1, j] - w[i-1, j])/(2*h)
    
def dwdy_central(w, h, i, j):
    """calculates dw/dy[i,j] using central difference and 2h interval"""
    return (w[i, j+1] - w[i, j-1])/(2*h)

def dwdx_backward(w,h, i, j): 
    """calculates dw/dy[i, j] using backward difference and h interval"""
    return (w[i, j] - w[i-1, j])/h

def dwdy_backward(w, h, i, j): 
    """calculate dw/dx[i, j] using backward difference and h interval """
    return (w[i, j] - w[i, j -1])/h 
    
def dwdy_forward(w, h, i, j):
    """"calculate dw/dy[i, j] using forward difference and h interval"""
    return (w[i, j + 1] - w[i, j])/h 

def dwdx_forward(w, h, i, j):
    """calculate dw/dx[i, j] using forward difference and h interval"""
    return (w[i + 1, j] - w[i, j])/h 


def Intensity(phi, dwdx, dwdy):
    """Calculates the intensity of illumination for scaler inputs as per 
    Equation in textbook.
    """
    denominator = np.cos(phi) * dwdx + np.sin(phi) * dwdy 
    numerator = np.sqrt((dwdx**2) + (dwdy)**2 + 1)
    return -denominator/numerator
    

f = open('N46E006.hgt', 'rb')

n = 1201 # resolution
w = np.zeros(n * n )
h = 420
phi = np.pi/6 #for intensity function 
I = np.zeros((n, n)) #empty intensity array 
dwdx = np.zeros((n, n))
dwdy = np.zeros((n, n))

# Populating w array
w = np.zeros([n,n])
for i in range(n): 
    for j in range(n):
        buf = f.read(2)
        w[i,j] = struct.unpack('>h', buf)[0]




plt.rc('text', usetex=True)             # use LaTeX for text
plt.rc('font', family='serif')          # use serif font
plt.rcParams.update({'font.size': 15})  # set font size 


plt.figure(figsize=(10,10)) # fig size
plt.title("w(x,y), from N46E006.hgt")
plt.imshow(w, extent=[6,7,46,47], vmin=400, cmap='gist_gray') # plot w, with vmin = 300
plt.colorbar(label='w(x,y) in metres', shrink=0.7,
             orientation='vertical', pad=0.3)
plt.ylabel("Latitude $^{\circ}$N")
plt.xlabel('Longtitude $^{\circ}$E')
plt.savefig('Q3_alt.pdf')
plt.show()


# populatin dwdx and dwdx array
for i in range(n):
    for j in range(n): 
        # dw/dx computations
        if i == 0: # the first row 
            dwdx[i,j] = dwdx_forward(w, h, i, j)
        elif i  == n - 1: # the last row
            dwdx[i, j] = dwdx_backward(w, h, i, j)
        else: # centre rows
            dwdx[i, j] = dwdx_central(w, h, i, j)
        # dw/dy computations
        if j == 0: # first column
            dwdy[i, j] = dwdy_forward(w, h, i, j)
        elif j == n - 1: # last column
            dwdy[i, j] = dwdy_backward(w, h, i, j)
        else: # centre columns
            dwdy[i, j] = dwdy_central(w, h, i, j)  

# Calculating intensitt
for i in range(n):
    for j in range(n): 
        I[i,j] = Intensity(phi, dwdx[i,j], dwdy[i,j])
        

 
# Plotting intensity on the grid
plt.figure(figsize=(10,10))
plt.title('Intensity of illumination from N46E006.hgt')
plt.imshow(I, extent=[6,7,46,47], vmin=-0.05, vmax = 0.05, cmap='gist_gray')
plt.colorbar(label='Intensity', shrink=0.7,
             orientation='vertical', pad=0.3)
plt.ylabel("Latitude $^{\circ}$N")
plt.xlabel('Longtitude $^{\circ}$E')
plt.savefig('Q3_Intensity.pdf')
plt.show()


        