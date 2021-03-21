"""
Lab08 PHY407
@author: Genevieve Beauregard(primarily), Anna Shirkalina 
Q2
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def F1(u, eta, g): 
    """From vector function derived in Equation 7 of handout. 
    Takes in scalar u and eta, outputs first component F"""
    
    return 0.5*(u**2) + g*eta

def F2(u, eta, eta_b):
    """From vector function derived in Equation 7.
    Takes in scalar u and eta, outputs the second compoenent."""

    return (eta - eta_b)*u


# we calculate eta0 <average> part for the gaussain
def gauss(x, A, mu, sigma): 
    """Returns $A exp(-(x - mu)^2/sigma^2) underintegral for averaging
    Input: A, mu, sigma floats
            x array-like
    Output: $A exp(-(x- mu)^2/sigma^2)$ 
    """
    return A* np.exp(-(x - mu)**2 / sigma**2)

# Initialize constants 
L = 1 #length
J = 50 # no of points in grif
dx = L/J # [m] spatial step size
x_array = np.linspace(0, L, J+1) #spatial grid to match x = 0 to x= 1

dt = 0.01 # [s] time step size
maxtime = 6.0
N_timesteps = int(maxtime // dt) # no of time steps 
t_array = np.arange(0, maxtime, dt) # time grid


g = 9.81  #[m/s^2] gravitational acceleration
eta_b = 0 # [m] bottom topography, flat
H = 0.01 # [m] avaerage water column height
A = 0.002 # [m]
mu = 0.5 # [m]
sigma = 0.05 # [m]


# calculate average (TA Alex told us this was alright)
average_gauss = np.mean(gauss(x_array, A, mu, sigma))

# setting the initial coniditions for n.
initialeta_0 = H + gauss(x_array, A, mu, sigma) - average_gauss

# initial u array
initalu_0 = np.zeros(J+1, dtype=float)

eta = initialeta_0 # i = 0
u = initalu_0 #
u_new = np.zeros(len(u))
eta_new = np.zeros(len(eta))
# loop over time and space

for i in range(len(t_array)):
    
    
    for j in range(len(u)):

        
        # # had these if statements initially, moved them outside, but i dont know 
        # # if correct to move them out 
        if j ==0: # forward difference scheme
            u_new[j]= u[j] # boundary
            eta_new[j] = eta[j] - (dt/(dx))*(F2(u[j+1], eta[j+1], eta_b) -
                                                F2(u[j], eta[j], eta_b))
        
         
                   
        elif j==J: # backward different scheme
            u_new[j] = u[j] #boundary
            eta_new[j] = eta[j] - (dt/(dx))*(F2(u[j], eta[j], eta_b) -
                                                F2(u[j-1], eta[j-1], eta_b))

        
        else: 
            u_new[j] = u[j] - (dt/(2*dx)) * (F1(u[j+1],eta[j+1],g)
                                              - F1(u[j-1],eta[j-1],g))
            
            eta_new[j] = eta[j] - (dt/(2*dx))*(F2(u[j+1], eta[j+1], eta_b) -
                                                F2(u[j-1], eta[j-1], eta_b))
    


    plt.clf() # clear the plot
    plt.title('t ='+str(t_array[i]))
    plt.plot(x_array, eta) 
    plt.xlabel('x, [m]')
    plt.ylabel('$\eta$ [m]')
    plt.draw()
    plt.pause(0.01)
        
    
    # # animation breaks on my computer 
    if t_array[i]== 1.0 or t_array[i]==0.0 or t_array[i] == 4.0:
        plt.clf() # clear the plot
        plt.title('t ='+str(t_array[i]))
        plt.plot(x_array, eta) 
        plt.xlabel('x, [m]')
        plt.ylabel('$\eta$ [m]')
        plt.draw()
        plt.pause(0.01)
        
        # plt.clf()
        # plt.title('t ='+str(round(t_array[i],2)))
        # plt.plot(x_array, u) 
        # plt.xlabel('x, [m]')
        # plt.ylabel('$u$ [m]')
        # plt.draw()
        # plt.pause(0.01) #pause to allow a smooth animation

    eta = np.copy(eta_new)
    u = np.copy(u_new)
    

    
        





    

    
