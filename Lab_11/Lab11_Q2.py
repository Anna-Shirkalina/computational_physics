"""
Lab11 PHY407
@author: Genevieve Beauregard(primarily), Anna Shirkalina 
Q2 Adapted from L11-Ising1D-start.py to a 2D case with animation. 
"""


# import modules
import numpy as np
from random import random, randrange
import matplotlib.pyplot as plt
import matplotlib.animation as animation



def energyfunction(J_, dipoles):
    """ function to calculate energy """
    # adjacent row-wise 
    row_multiply = np.sum(dipoles[0:-1]*dipoles[1:])
    
    # adjacent column-wise
    column_multiply = np.sum(dipoles[:,0:-1]*dipoles[:,1:])
    
    energy = -J_*(column_multiply + row_multiply)
    return energy



def acceptance(Enew, E,kB,T):
    """ Function for acceptance probability """
    result = False 
    
    p = np.exp(-(Enew-E)/(kB*T))
    if Enew - E <= 0:
        result = True 
    elif Enew - E > 0 and p > random(): 
        result = True

    return result  # result is True of False


# define constants
kB = 1.0
T = 3.0
J = 1.0
N = 20 # length of side of array 
steps = 100000 # no of steps 


# generate array of dipoles and initialize diagnostic quantities
# create random array of +1/-1 
dipoles_initial = np.zeros((N, N), int)
for i in range(N): 
    for j in range(N): 
        dipoles_initial[i][j] = randrange(-1,2,2) #creates a random array of +1/-1

dipoles = np.copy(dipoles_initial)
energy = []  # empty list; to add to it, use energy.append(value)
magnet = []  # empty list; to add to it, use magnet.append(value)
dipoles_stack = []

E = energyfunction(J, dipoles)
energy.append(E)
magnet.append(np.sum(dipoles))
dipoles_stack.append(np.copy(dipoles))



# Metropolis implement
for i in range(steps):
    
    xcoord = randrange(N) # choose victim
    ycoord = randrange(N)
    dipoles[xcoord, ycoord] *= -1  # propose to flip the victim
    Enew = energyfunction(J, dipoles)  # compute Energy of proposed new state

    # calculate acceptance probability
    accepted = acceptance(Enew, E, kB, T)
    
    # Accept or keep if needed
    if accepted:
        energy.append(Enew)
        E = np.copy(Enew) 
        magnet.append(np.sum(dipoles))
        dipoles_stack.append(np.copy(dipoles))
    else: 
        energy.append(E)
        dipoles[xcoord,ycoord] *= -1 #flip back 
        magnet.append(np.sum(dipoles))
        dipoles_stack.append(np.copy(dipoles)) # np.copy is NECESSARY
        # Suspect dipoles[xcoord,ycoord] flips act on the 
        # entries of the list otherwise



# # ANIMIATION 
# Referred to: http://jakevdp.github.io/blog/2012/08/18/matplotlib-animation-tutorial/
# https://stackoverflow.com/questions/17212722/matplotlib-imshow-how-to-animate
# There was a failed attempt at using ArtistAnimation, which is far more 
# intuitive for arrays we already have stored. But the colourbar buggy. 


fig = plt.figure()
a = dipoles_stack[0]
im = plt.imshow(a)

# skip every 100 to save time, the dipole list is long!
skip_frames_every = 100
included_frames = np.arange(0,steps + skip_frames_every,skip_frames_every) 

def animate_func(i):
    im.set_array(dipoles_stack[i])
    return [im]

anim = animation.FuncAnimation(fig, animate_func,\
                               frames=included_frames\
                                   , blit=True)
plt.xlabel('x coordinates')
plt.ylabel('y coordinates')
plt.colorbar()
plt.title('Spin Evolution, T={}'.format(T))
anim.save('Q2c_AnimationT{}.mp4'.format(T), writer='ffmpeg')

print('Done!')



plt.rc('text', usetex=True)             # use LaTeX for text
plt.rc('font', family='serif')          # use serif font
plt.rcParams.update({'font.size': 17})  # set font size 


# plot energy, last dipole & magnetization
plt.figure()
plt.plot(np.array(energy))
plt.xlabel('Steps')
plt.ylabel('Energy scaled Joules')
plt.title('Energy over steps for Ising Model T={}'.format(T))
plt.grid()
plt.tight_layout()
plt.savefig('Q2_energy_T_{}.pdf'.format(T))

plt.figure()
plt.plot(np.array(magnet))
plt.xlabel('Steps')
plt.ylabel('Magnetization')
plt.grid()
plt.title('Magnetization Ising Model T={}'.format(T))
plt.tight_layout()
plt.savefig('Q2_magnetizationT_{}.pdf'.format(T))





