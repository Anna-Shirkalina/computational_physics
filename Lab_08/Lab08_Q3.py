"""
Lab08 PHY407
@author: Genevieve Beauregard, Anna Shirkalina(primarily)
Q3
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def F1(u, eta, g):
    """From vector function derived in Equation 7 of handout.
    Takes in scalar u and eta, outputs first component F"""

    return 0.5 * (u ** 2) + g * eta


def F2(u, eta, eta_b):
    """From vector function derived in Equation 7.
    Takes in scalar u and eta, outputs the second components."""

    return (eta - eta_b) * u


# we calculate eta0 <average> part for the gaussain
def gauss(x, A, mu, sigma):
    """Returns $A exp(-(x - mu)^2/sigma^2) under integral for averaging
    Input: A, mu, sigma floats
            x array-like
    Output: $A exp(-(x- mu)^2/sigma^2)$
    """
    return A * np.exp(-(x - mu) ** 2 / sigma ** 2)


def eta_b_initilization(a, b, N):
    """Return the bottom array topography as specified in the lab mannual
    eta_b = n_bs * (1 + tanh((x - x0) * alpha)
    a and b are the x values in m of the boundaries of the surface, N is the
    total number of divisions in the x array
    """
    H = 0.01
    n_bs = H - 4e-4
    alpha = 8 * np.pi
    x_0 = 0.5
    x_array = np.linspace(a, b, N)
    return 0.5 * n_bs * (1 + np.tanh((x_array - x_0)*alpha))

print("Are you solving 3a or 3b? Type in kernel [a/b]")

question = input()

if question == 'a': 
    # Constants for 3a 
    J = 50  # no of points in grid
    A = 0.002  # [m]
    mu = 0.5  # [m]
    sigma = 0.1  # [m]
    L = 1  # length
    eta_b = np.zeros(J + 1, dtype=float)
    dt = 0.01  # [s] time step size
    maxtime = 4.5
    N_timesteps = int(maxtime // dt)  # no of time steps
    t_array = np.arange(0, maxtime, dt)  # time grid
    
elif question == 'b': 
    # Constants for 3b
    J = 150  # no of points in grid
    A = 0.0002  # [m]
    mu = 0.0  # [m]
    sigma = 0.1  # [m]
    L = 1  # length
    eta_b = eta_b_initilization(0, L, J + 1)  # [m] bottom topography, flat
    dt = 0.001  # [s] time step size
    maxtime = 5.0 + dt #5000 iterations :)
    N_timesteps = int(maxtime // dt)  # no of time steps
    t_array = np.arange(0, maxtime, dt)  # time grid
else: 
    print('Please rerun the code and give a valid input in kernel.')
    print('Type either a or b.')

# Initialize the rest of the constants
dx = L / J  # [m] spatial step size
x_array = np.linspace(0, L, J + 1)  # spatial grid to match x = 0 to x= 1


g = 9.81  # [m/s^2] gravitational acceleration
H = 0.01  # [m] average water column height

# calculate average (TA Alex told us this was alright)
average_gauss = np.mean(gauss(x_array, A, mu, sigma))

# setting the initial conditions for n.
initial_eta_0 = H + gauss(x_array, A, mu, sigma) - average_gauss

# initial u array
initial_u_0 = np.zeros(J + 1, dtype=float)

eta = initial_eta_0  # i = 0
u = initial_u_0  #
u_new = np.zeros(len(u))
u_half = np.zeros(len(u))
eta_new = np.zeros(len(eta))
eta_half = np.zeros(len(eta))
# loop over time and space

for i in range(len(t_array)):
    for j in range(len(u)):

        if j == 0:
            u_new[j] = 0.0  # boundary
            eta_new[j] = eta[j] - (dt / (dx)) * (
                        F2(u[j + 1], eta[j + 1], eta_b[j+1]) -
                        F2(u[j], eta[j], eta_b[j]))
            u_half[j] = 0.5 * (u[j + 1] + u[j]) - (dt / (2 * dx)) \
                        * (F1(u[j + 1], eta[j + 1], g) -
                            F1(u[j], eta[j], g))
            eta_half[j] = 0.5 * (eta[j + 1] + eta[j]) - (dt / (2 * dx)) * \
                          (F2(u[j + 1], eta[j + 1], eta_b[j+1]) -
                            F2(u[j], eta[j], eta_b[j]))

        elif j == J:  # backward different scheme
            u_new[j] = 0.0  # boundary
            eta_new[j] = eta[j] - (dt / (dx)) * (F2(u[j], eta[j], eta_b[j]) -
                                                  F2(u[j - 1], eta[j - 1],
                                                    eta_b[j-1]))

        else:
            u_half[j] = 0.5 * (u[j + 1] + u[j]) - (dt / (2*dx))\
                        * (F1(u[j+1], eta[j+1], g) -
                            F1(u[j], eta[j], g))

            eta_half[j] = 0.5 * (eta[j+1] + eta[j]) - (dt / (2*dx)) * \
                            (F2(u[j+1], eta[j+1], eta_b[j]) -
                            F2(u[j], eta[j], eta_b[j]))

            u_new[j] = u[j] - (dt / dx) * (F1(u_half[j], eta_half[j], g) -
                                                F1(u_half[j-1],
                                                  eta_half[j-1], g))
            eta_b_plus_half = 0.5 * (eta_b[j] + eta_b[j + 1])
            eta_b_minus_half = 0.5 * (eta_b[j - 1] + eta_b[j])
            eta_new[j] = eta[j] - (dt / dx) * (
                        F2(u_half[j], eta_half[j], eta_b_plus_half) -
                        F2(u_half[j-1], eta_half[j-1], eta_b_minus_half))


    # animation
    plt.clf() # clear the plot
    plt.title('t ='+str(round(t_array[i],4))+'s, Lax Wendroff')
    # if question == 'a':
    #     plt.ylim(-0.001, 0.013) # appropriate fixed axis for Q3a 
    plt.plot(x_array, eta, label='$\eta$')
    plt.plot(x_array, eta_b, label='$\eta_b$') #UNCOMMENT for 3b simulation
    plt.xlabel('x [m]')
    plt.ylabel('$\eta$ [m]')
    if question == 'a':
        plt.ylim(-0.001, 0.013) # appropriate fixed axis for Q3a 
    else:
        plt.ylim(0.00950, 0.01025)
    plt.legend()
    plt.grid()
    # if t_array[i] == 0.0 or t_array[i] == 1.0 or t_array[i] == 2.0 or\
    #     t_array[i] == 4.0:
    #     plt.savefig('Q3{}_bothplot_t={}.pdf'.format(question, str(t_array[i])))
    plt.draw()
    plt.pause(0.01)
    
    # # # to save plots of the free surface of the water, will run slow(!)
    plt.rc('text', usetex=True)             # use LaTeX for text
    plt.rc('font', family='serif')          # use serif font
    plt.rcParams.update({'font.size': 15})  # set font size 
    if t_array[i] == 0.0 or t_array[i] == 1.0 or t_array[i] == 2.0 or\
        t_array[i] == 4.0:
        plt.clf()  # clear the plot
        plt.title('t =' + str(t_array[i])+'s, Lax-Wendroff Tsunami')
        plt.plot(x_array, eta, label='$\eta$')
        #plt.plot(x_array, eta_b, label='$\eta_b$')
        plt.xlabel('x, [m]')
        plt.ylabel('Ocean Height [m]')
        #plt.ylim(0.00950)
        plt.legend()
        plt.tight_layout()
        plt.grid()
        plt.savefig('Q3{}_t='.format(question) + str(int(t_array[i])) +\
                    '_eta_Lax_Wendroff.pdf')
        plt.draw()
        plt.pause(0.01)


    eta = np.copy(eta_new)
    u = np.copy(u_new)










