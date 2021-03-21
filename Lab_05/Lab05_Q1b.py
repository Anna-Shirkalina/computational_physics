"""
Lab5 PHY407
@author: Genevieve Beauregard(primarily), Anna Shirkalina 
Q1b
"""

def filterSettoZero(y, k):
    """Filters a time series y by setting its all but the first k% of its coeff
    to be zero, using rfft routine from numpy.
    Input: 
        time series array y
        percentage of first coefficients not set to zero, k
    Output:
        Filtered time series
        """
        
    c = fft.rfft(y) # obtain the index after which we set coeff to zero
    N = len(c) #obtain length
    N_cutoff = int(k*0.01 * N)
    c[N_cutoff:] = 0 # setting to zero
    y_filtered = fft.irfft(c) #return to time series
    
    return y_filtered


import numpy as np 
import numpy.fft as fft 
import matplotlib.pyplot as plt

# Loading data
dow = np.loadtxt("dow.txt", float)

# configuring plots, rica/erik taught me this snippet to format fonts
plt.rc('text', usetex=True)             # use LaTeX for text
plt.rc('font', family='serif')          # use serif font
plt.rcParams.update({'font.size': 15})  # set font size 

# plotting function of time
plt.figure(figsize=(10,5))
plt.plot(dow)
plt.xlabel("Time (days)")
plt.ylabel("Price")
plt.title("Daily closing values of the Dow from 2006 to 2010")
plt.savefig("Lab05_Q1bi")


# setting te coeff of the fourier transform to be zero we filter the time series
dow_filtered_10 = filterSettoZero(dow, 10)
dow_filtered_2 = filterSettoZero(dow, 2)

# Now we plot
plt.figure(figsize=(10,5))
plt.plot(dow, label='Original', alpha=0.7)
plt.plot(dow_filtered_10, label='Filtered, 10 \%', )
plt.plot(dow_filtered_2, label='Filtered, 2 \%' )
plt.legend()
plt.xlabel("Time (days)")
plt.ylabel("Price")
plt.title("Daily closing values of the Dow Jones Index from 2006 to 2010")
plt.savefig("Lab05_Q1bi_filtered.pdf")