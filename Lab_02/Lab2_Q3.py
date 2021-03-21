"""
@author: Anna Shirkalina (primarily), Genevive Beauregard
"""

import numpy as np
import scipy.optimize as spop
import matplotlib.pyplot as plt
from scipy import special
import Lab2_functions as func

# initialize parameters globably
separation = 20 * 1e-6  # in m
wave_length = 500 * 1e-9  # m
focal_length = 1  # m
x_screen_width = 0.1  # m
width_diffraction_grating_1 = 20e-6 * 10 # the width for Q3 a -e i)
width_diffraction_grating_2 = 90e-6 # the width of grating for e ii)
width_diffraction_grating = width_diffraction_grating_2


def q(u: float):
    """Return the intensity of the transmission function at a distance u form
    central axis. Uncomment the blocks for the relevant q function.
    @:param separation the distance between the slits in micrometers
    """
    # transmission profile for Q3a-d
    # alpha = np.pi / separation
    alpha = np.pi / separation

    # # transmission profile for Q3a-d
    # return (np.sin(alpha * u)) ** 2

    # # transmission profile for Q3 e i)
    # return ((np.sin(alpha * u)) ** 2) * (np.sin(alpha * u / 2)) ** 2

    # transmission profile for Q3 ii)
    if -30e-6 >= u >= -40e-6:
        return 1
    if 30e-6 <= u <= 50e-6:
        return 1
    else:
        return 0


def diffraction_pattern_integrand(u: float, screen_position: float):
    """Return the value of the integrand for the intensity function
    @:param u the location of the from the central axis
    @:param x, the position on the axis
    """
    return np.sqrt(q(u)) * np.exp(2j * np.pi * u * screen_position /
                                  (wave_length * focal_length))


# integrate the intensity function
n = 1000  # horizontal resolution, number of horizontal slices
diffraction_pattern = np.zeros(n)  # initialize the array
i = 0  # index counter for adding to the array
N = 100 #integration slices

# want to compute the integral along the size of the screen
for x in np.linspace(-x_screen_width/2, x_screen_width/2, n):

    # use the Simon's integration method to compute the integral
    intensity_diffraction = func.simp(-width_diffraction_grating/2,
                                      width_diffraction_grating/2, N,
                                      lambda u:
                                      diffraction_pattern_integrand(u, x))
    # take the absolute value of the integral
    diffraction_pattern[i] = np.abs(intensity_diffraction)**2
    i += 1

# We want to copy the array of intensities along the y- axis
stretched_pattern = diffraction_pattern[np.newaxis, :] * np.ones((100, 1))

# configuring plots, rica/erik taught us this snippet to format fonts
#plt.rc('text', usetex=True)             # use LaTeX for text
plt.rc('font', family='serif')          # use serif font
plt.rcParams.update({'font.size': 14})  # set font size

plt.figure(1)
plt.imshow(stretched_pattern, extent=[0, x_screen_width * 100, 0, 1],\
           cmap='hot', vmax=1e-9)
plt.xlabel("Distance in cm along the screen")
plt.yticks([])

plt.colorbar(label='Light intensity in $\\frac{W}{m^2}$', shrink=0.7,
             orientation='horizontal', pad=0.3)
plt.show()


### code to generate graph of the transmission function for 3e
def q_piecewise(u):
    if -30e-6 >= u >= -40e-6:
        return 1
    if 30e-6 <= u <= 50e-6:
        return 1
    else:
        return 0


y = np.arange(-50e-6, 60e-6, 1e-6)
p_y = []

for i in y:
    p_y.append(q_piecewise(i))

plt.figure(2)
plt.plot(y, p_y)
plt.xlabel("Position on the grating")
plt.ylabel("Transmission function")
plt.show()
