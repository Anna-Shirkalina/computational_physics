"""
Lab7 Q2b PHY407
@author: Genevieve Beauregard(primarily), Anna Shirkalina 

This is heavily adapted from the solution provided by NG.
I kept some of his comments and his constants.
"""

from scipy.constants import G, au  
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from time import time


def f(r_):
    """ Right-hand-side for velocity equation: gravitational force
    INPUT: r_ = [x, y, vx, vy]
    OUTPUT: [vx, vy, Fx, Fy]: along-x and -y graviational forces
    """
    x, y, vx, vy = r_[0], r_[1], r_[2], r_[3]
    M = 1.9891e30  # [kg] Mass of the Sun
    r = (x**2 + y**2)**.5  # [m] distance to Sun
    prefac = -M*G/r**3
    return np.array([vx, vy, x*prefac, y*prefac], float)


ftsz = 16
font = {'family': 'normal', 'size': ftsz}  # font size
rc('font', **font)


delta = 1000/(365.25*24*3600) # [m]/s target accuracy 1km/year 

M = 1.9891e30  # [kg] Mass of the Sun
PH = 4.4368e12  # [m] perihelion of Pluto in question
VP = 6.1218e3  # [m/s] velocity of Pluto at perihelion
h = 3600.  # [s] time step

Nrevs = 5  # []  # of revolutions around the Sun
year = 248 * 365.25*24*3600.  # [s] duration of a pluto orbit
T = Nrevs*year  # [s] duration of integration
Hweeks = 170 # weeks for hstep
H = Hweeks * 7 * 24 * 60 * 60 # [s] 'bigstep' 1 week
Nsteps = int(T/H)  # [] number of time steps



# I define the x, y axes as: Earth start at perielion, along x>0, with along-y
# positive velocity
# initial condition 



v0 = [0., VP]  
r0 = [PH,0.]
r = np.array([r0[0], r0[1], v0[0], v0[1]])


#initializing arrays 
tpoints = np.arange(0, T, H)
xpos =[]
ypos =[]
xvel =[]
yvel =[]


# we use this to determine the speed of the loop
start = time()

# BS method  -----------------------------------------------------------------|
# Based on bulirsch.py


# Do the "big steps" of size H
for t in tpoints:
    # update all your positions
    xpos.append(r[0])
    ypos.append(r[1])
    xvel.append(r[2])
    yvel.append(r[3])
    # Do one modified midpoint step to get things started
    n = 1
    r1 = r + 0.5*H*f(r)
    r2 = r + H*f(r1)

    # The array R1 stores the first row of the
    # extrapolation table, which contains only the single
    # modified midpoint estimate of the solution at the
    # end of the interval
    R1 = np.zeros([1, len(r)], float)
    R1[0] = 0.5*(r1 + r2 + 0.5*H*f(r2))

    # Now increase n until the required accuracy is reached
    error = 2*H*delta
    while error > H*delta:

        n += 1
        h = H/n

        # Modified midpoint method
        r1 = r + 0.5*h*f(r)
        r2 = r + h*f(r1)
        for i in range(n-1):
            r1 += h*f(r2)
            r2 += h*f(r1)

        # Calculate extrapolation estimates.  Arrays R1 and R2
        # hold the two most recent lines of the table
        R2 = R1
        R1 = np.zeros([n, len(r)], float)
        R1[0] = 0.5*(r1 + r2 + 0.5*h*f(r2))
        for m in range(1, n):
            epsilon = (R1[m-1]-R2[m-1])/((n/(n-1))**(2*m)-1)
            R1[m] = R1[m-1] + epsilon
        error = abs(epsilon[0])

    # Set r equal to the most accurate estimate we have,
    # before moving on to the next big step
    r = R1[n-1]

end = time()

print('The time it takes to run the loop is, '\
      + str(round(end - start, 2)) +'s with h step of '+ str(Hweeks)+' weeks')        

# Plot -----------------------------------------------------------------------|
# convert to Au
xpos = np.array(xpos)
xpos_AU = xpos/au

ypos = np.array(ypos)
ypos_AU = ypos/au



plt.figure(dpi=100)
plt.plot(xpos_AU, ypos_AU, 'k.')
plt.axvline(0.)
plt.axhline(0.)
plt.title('Orbit of the Pluto around the Sun')
plt.grid()
plt.xlabel('$x$ (AU)')
plt.ylabel('$y$ (AU)')
plt.axis('equal')
plt.tight_layout()
plt.savefig('Question4b.pdf')
plt.show()