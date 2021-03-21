"""
Lab5 PHY407
@author: Genevieve Beauregard(primarily), Anna Shirkalina 
Q1d
"""

import numpy as np 
import numpy.fft as fft 
import matplotlib.pyplot as plt



def PlotTimeAndFreq(y, name:str): 
    """Plots the first 10000 points of audio time signal y and its frequency 
    spectrum. Name will be used to label the plots and files appropriately. 
    Input: 
        y time signal, array 
        name, string
    Output: 
        frequency axis, array
        |c|^2, array
    """
        
    #select first 10000
    y_slice = y[:10000]
    
    # create time array
    dt = 1/44100 
    time = np.arange(0, len(y_slice)*dt, dt)

    #Plot time signal
    plt.figure(figsize=(10,5))
    plt.plot(time, y_slice, label='trumpet')
    plt.xlabel("Time(s)")
    plt.ylabel("Magnitude")
    plt.title("Waveform of the {}.txt".format(name))
    plt.savefig("Lab05_Q1d_{}time.pdf".format(name))
    
    #take fast fourier transform, we use appropriate shifts
    c = fft.fftshift(fft.fft(y))
    
    #create frequency array
    cfreq = fft.fftshift(fft.fftfreq(len(c), dt))

    #plot absolute value
    plt.figure(figsize=(10,5))
    plt.plot(cfreq, abs(c)**2, label='trumpet')
    plt.xlabel("Freqauency(Hz)")
    plt.ylabel("$|c|^2$")
    plt.title("Power Spectrum of the {}.txt".format(name))
    plt.xlim(0, 5000)
    plt.savefig("Lab05_Q1d_{}freq.pdf".format(name))
    
    
    return cfreq, c**2



# load data into arrays
piano = np.loadtxt("piano.txt", float)
trumpet = np.loadtxt("trumpet.txt", float)

# configuring plots, rica/erik taught me this snippet to format fonts
plt.rc('text', usetex=True)             # use LaTeX for text
plt.rc('font', family='serif')          # use serif font
plt.rcParams.update({'font.size': 15})  # set font size 


# Plot and obtain fourier transform data
freq_trumpet, trumpet_csquare = PlotTimeAndFreq(trumpet, 'trumpet')
freq_piano, piano_csquare = PlotTimeAndFreq(piano,'piano')


print("We obtain the frequency for which $|c|^2$ is the largest")

# obtain index for which the maximum magnitude in the power spectrum occurs
trumpfreq_max = np.argmax(trumpet_csquare)
# print the relevant frequency
print("The frequency for the trumpet is "\
      +str(abs(freq_trumpet[trumpfreq_max]))+"Hz")
    

# obtain index for which the maximum magnitude in the power spectrum occurs
pianofreq_max = np.argmax(piano_csquare)

# print the relevant frequency
print("The frequency for the piano is "\
      +str(abs(freq_piano[pianofreq_max]))+"Hz")

    