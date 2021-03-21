"""
Lab5 PHY407
@author: Genevieve Beauregard(primarily), Anna Shirkalina 
Q1c
"""
import numpy as np 
import numpy.fft as fft 
import matplotlib.pyplot as plt
from Lab05_Q1b import filterSettoZero
import dcst 



# load data into arrays
dow2 = np.loadtxt("dow2.txt", float)

# filter with 2%
dow2_filtered = filterSettoZero(dow2, 2)


# plotting function of time
plt.figure(figsize=(10,5))
plt.plot(dow2, label='Original', alpha=0.7)
plt.plot(dow2_filtered, label='Filtered, 2 \%' )
plt.legend()
plt.xlabel("Time (days)")
plt.ylabel("Price")
plt.title("Daily closing values of the Dow Jones Index from 2004 to 2008")
plt.savefig("Lab05_Q1ci")


# we repeat the process but with the discrete cosine routine in dcst.py

c_dow2_cos = dcst.dct(dow2)
N = len(c_dow2_cos)
c_dow2_cos[int(N* 0.02):] = 0
dow2_cosfiltered = dcst.idct(c_dow2_cos)


plt.figure(figsize=(10,5))
plt.plot(dow2, label='Original', alpha=0.7)
plt.plot(dow2_cosfiltered, label='Filtered DCT, 2 \%' )
plt.legend()
plt.xlabel("Time (days)")
plt.ylabel("Price")
plt.title("Daily closing values of the Dow Jones Index from 2004 to 2008")
plt.savefig("Lab05_Q1cii")



