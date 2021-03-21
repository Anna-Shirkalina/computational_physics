"""
Lab4 PHY407
@author: Genevieve Beauregard(primarily), Anna Shirkalina 
Q1a
"""

import numpy as np 
import matplotlib.pyplot as plt


# loading data into arrays
y = np.loadtxt("sunspots.txt", float)
time = y[:,0]
sunspot = y[:,1]




# configuring plots, rica/erik taught me this snippet to format fonts
plt.rc('text', usetex=True)             # use LaTeX for text
plt.rc('font', family='serif')          # use serif font
plt.rcParams.update({'font.size': 15})  # set font size 

# plotting function of time
plt.figure(figsize=(10,5))
plt.plot(time,sunspot)
plt.xlabel("Time (Months)")
plt.ylabel("Number of Sunspots")
plt.title("Observed number of sunspots since Jan 1749")
plt.savefig("Lab05_Q1ai")



# obtaining the c_abs function
c_abssquare =np.abs(np.fft.rfft(sunspot))**2

# obtaining frequency 
c_freq = np.fft.rfftfreq(len(time))

plt.figure(figsize=(10,5))
plt.plot(c_freq, c_abssquare)
plt.xlabel("Frequency (/month)")
plt.xlim(0,0.01)
plt.ylabel("$|c_k|^2$")
plt.title("Power Spectrum of Observed Sunspots")
plt.savefig("Lab05_Q1aii")





