"""
Lab7 PHY407
@author: Genevieve Beauregard(primarily), Anna Shirkalina 
Q2d
We basically adapt 2c but for a varying omega 
"""

import numpy as np
import matplotlib.pyplot as plt
from progressbar import ProgressBar
from Lab09_Q2b import dst2, idst2, dct2, idct2, dsct2, idsct2, dcst2, idcst2
pbar = ProgressBar()


def J(t, omega, J_0sin):
    """ Implementation of Equation 8
    Input: 
        J_0sin, time independent coefficient P x P spatial grid
        omega, frequency scalar 
        t, time scalar
        
    Output: 
        J_z Px P grid for a given time t
    """
    return J_0sin * np.sin(omega*t)

def ObtainallFourier(E_z, H_x, H_y, J_z): 
    """Obtains all the fourier coefficients for spatial grids E_z, H_y, H_z, 
    J_z, using their respective discrete sin/cos 2d routines, as per Equation 
    10.
    Inputs: 
        E_z, P x P array 
        H_y, P x P array 
        H_z, P x P array
        J_z, P x P array
    Outputs: 
        Fourier coefficients E_z,H_y,H_z and J_z in P+1 x P+1 arrays. 
    """
    fourierE_z = dst2(E_z)
    X = dsct2(H_x) # notation in lab handout: X is fourier coeff of H_x
    Y = dcst2(H_y) # likewise
    fourierJ_z = dst2(J_z)
    
    return fourierE_z, X, Y, fourierJ_z

def ObtainallFourierInverse(fourierE_z, X, Y):
    """Obtains inverse of spatial grids fourierE_z, fourierH_y, fourierH_z. 
    """
    
    E_z = idst2(fourierE_z)
    H_x = idsct2(X) # as per handout notation Eqn 10
    H_y = idcst2(Y) 
    #J_z = idst2(fourierJ_z)
    
    return E_z, H_x, H_y, #J_z


def Eqn11a(fourierE_z, X, Y, fourierJ_z, D_x, D_y, tau, p,q): 
    """Eqn 11a implementation. Note all this for scalar inputs values 
    at position p, q"""
    
    factor = (1 - (p**2)*(D_x)**2 - (q**2) * (D_y**2))
    numerator = (factor * fourierE_z + 2*q*D_y*X + 2*p*D_x * Y
                 + tau*fourierJ_z)
    denominator = 1 + (p**2 * D_x**2) + (q**2 * D_y**2)
    
    return numerator/denominator
    

def nextstepCrankNic(fourierE_z, X, Y, fourierJ_z, D_x, D_y, tau):
    """Obtains next step in crank nicholson for fourier coeff fourierE_z, X, 
    Y. Eqn 11 implementation
    Inputs: 
        fourierE_z, dst2 coef of E_z at current time step, P+1 x P+1 array
        X, fourier coeff of H_x at current time step, P+1 x P+1 array
        Y, same as above but H_y
        fourierJ_z, same as above but for J_z
        D_x, scalar 
        D_y, scalar
    Outputs: 
        Next time steps of fourierE_z, X,Y using equation 11.
    """
    shape = np.shape(fourierE_z) # obtain shape of grid
    nextfourierE_z = np.zeros(shape) # dummy arrays
    nextX = np.zeros(shape)
    nextY = np.zeros(shape)
    
    # loop over indices of arrays 
    for p in range(shape[0]):
        for q in range(shape[1]): 
            nextfourierE_z[p,q] = Eqn11a(fourierE_z[p,q], X[p,q], Y[p,q],\
                                          fourierJ_z[p,q], D_x, D_y, tau, p, q)
            
            nextX[p,q] = X[p,q] - q* D_y*(nextfourierE_z[p,q] + fourierE_z[p,q])
            
            nextY[p,q] = Y[p,q] - p * D_x * (nextfourierE_z[p,q] + fourierE_z[p,q])
            
    return nextfourierE_z, nextX, nextY


I_want_to_compute_everything = False

maxomega = 9.0
domega = 0.1 #feel free to change this 
omegas = np.arange(0, maxomega + domega, domega)


#Define constants 
T = 20 
tau = 0.01 
N = T/tau 

L_x = 1
L_y = 1
J_0 = 1 
m = 1 
n = 1
c = 1 
P = 32


# define 'subsidary' constants 

D_x = np.pi * c * tau /( 2*L_x)
D_y = np.pi * c * tau /( 2*L_y)

a_x = L_x/P
a_y = L_y/P

# Create space arrays 
x = np.linspace(0,L_x, P+1)
y = np.linspace(0, L_y, P+1)
xx, yy = np.meshgrid(x,y)

# Create initial J_z array, we can see that 
# J_z has a time independent part and a sin(omega t)time dependence
# We call this time independent part J_0sin
J_0sin = J_0 * np.sin(m * np.pi* xx/L_x) * np.sin(n * np.pi *yy/L_y)

# Now we deal with the time
# Create time array for looping 
t_array = np.arange(0, T+tau, tau)



# Loop over omegas, we use the if statement 

if I_want_to_compute_everything:
    
    E_zmaxarray = []
    
    
    for omega in pbar(omegas): 
        # Set initial H_x, H_y, E_z, J_0 for t = 0, these variables will 
        # updated in time loop
        H_x = np.zeros([P+1,P+1])
        H_y = np.zeros([P+1, P+1])
        E_z = np.zeros([P+1, P+1])
        J_z = J(0, omega, J_0sin)

        # Set the accumalator variables, we call it the stack, where the first 
        # index is time 
        H_xstack = np.array([H_x])
        H_ystack = np.array([H_y])
        E_zstack = np.array([E_z])
        J_zstack = np.array([J_z])
        
        for k in range(0, len(t_array)-1):
            # Obtain fourier coefficients of H_x, H_y, E_z, J_0 at time t_array[k] 
            fourierE_z, X, Y, fourierJ_z = ObtainallFourier(E_z, H_x, H_y, J_z)
    
            nextfourierE_z, nextX, nextY = nextstepCrankNic(fourierE_z, X, Y,\
                                                    fourierJ_z, D_x, D_y, tau)
    
            # next step spatial dim
            nextE_z, nextH_x, nextH_y = ObtainallFourierInverse(nextfourierE_z,\
                                                        nextX, nextY)
    
            nextJ_z = J(t_array[k+1], omega, J_0sin)
    
            # stack
            E_zstack = np.vstack((E_zstack, np.array([E_z])))
    
            # update variables 
            E_z = np.copy(nextE_z)
            H_x = np.copy(nextH_x)
            H_y = np.copy(nextH_y)
            J_z = np.copy(nextJ_z)
            
        # obtain max array entry at x = 0.5, y = 0.5 
        E_ztime = E_zstack[:,16,16] # obtain the one d time E(0.5, 0.5,t) 
        E_ztimeabs = np.abs(E_ztime) # taking abs for amplitude
        E_zmax = np.max(E_ztime) # taking maximum to find max amplitude 
        
        # collect 
        E_zmaxarray.append(E_zmax)
        
    np.savez('output_file', E_zmaxarray = E_zmaxarray)

else: 
    npzfile = np.load('output_file.npz')
    E_zmaxarray = npzfile['E_zmaxarray']
    
    
plt.rc('text', usetex=True)             # use LaTeX for text
plt.rc('font', family='serif')          # use serif font
plt.rcParams.update({'font.size': 15})  # set font size 

plt.figure()
plt.title('Maximum Amplitude of $E_z(0.5, 0.5, t)$ against $\omega$')
plt.plot(omegas, E_zmaxarray)
plt.xlabel('$\omega$')
plt.ylabel('amplitude $E_z(0.5,0.5,t)$')
plt.tight_layout()
plt.savefig('Q2d.pdf')
plt.show()

print(np.argmax(E_zmaxarray))
print(omegas[45])
    
        
        



