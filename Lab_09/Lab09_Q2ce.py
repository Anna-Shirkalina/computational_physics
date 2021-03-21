"""
Lab7 PHY407
@author: Genevieve Beauregard(primarily), Anna Shirkalina 
Q2c\e
"""

import numpy as np
import matplotlib.pyplot as plt
from Lab09_Q2b import dst2, idst2, dct2, idct2, dsct2, idsct2, dcst2, idcst2


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

def Omega_mn(c, L_x, L_y, m, n):
    """Returns omega as per Eqn 21 in the handout. The driving force.
    Input: 
    c, scaler
    L_x, L_y, length of box, scalar
    m, n moders, scalar
    Output:
    omega normal frequencies, scalar
    """
    return np.pi*c* np.sqrt(pow(n*L_x, -2) + pow(m*L_y, -2))

    
    

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


# This section is just to set omega. If we are 
# solving q2c, please type c into the kernel 
# solving q2e, please type e into the kernel
# the correct omega would be set

print("Which part are you solving? Type [c/e]")
question = input()

if question =='c':
    omega = 3.75 #omega for 2c
elif question =='e':
    omega = Omega_mn(c, L_x, L_y, m, n) #omega for driving force

else:
    print('Please type a valid input, either \'c\' or \'e\'.')

# Create space arrays 
x = np.linspace(0,L_x, P+1)
y = np.linspace(0, L_y, P+1)
xx, yy = np.meshgrid(x,y)

# Create initial J_z array, we can see that 
# J_z has a time independent part and a sin(omega t)time dependence
# We call this time independent part J_0sin
J_0sin = J_0 * np.sin(m * np.pi* xx/L_x) * np.sin(n * np.pi *yy/L_y)


# Set initial H_x, H_y, E_z, J_0 for t = 0, these variables will updated in 
# time loop
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


# Now we deal with the time
# Create time array for looping 
t_array = np.arange(0, T+tau, tau)


# Now that we have the time array we can begin looping from the first non-zero
# step for the  times to obtain the next information of t[k + 1]
# we use index k for time; p and q will be used exclusively for spatial grids
for k in range(0, len(t_array)-1):
    # Obtain fourier coefficients of H_x, H_y, E_z, J_0 at time t_array[k] 
    fourierE_z, X, Y, fourierJ_z = ObtainallFourier(E_z, H_x, H_y, J_z)
    
    nextfourierE_z, nextX, nextY = nextstepCrankNic(fourierE_z, X, Y,\
                                                    fourierJ_z, D_x, D_y, tau)
    
    # next step spatial dim
    nextE_z, nextH_x, nextH_y = ObtainallFourierInverse(nextfourierE_z,\
                                                        nextX, nextY)
    
    nextJ_z = J(t_array[k+1], omega, J_0sin)
    
    # stack the updated grid
    E_zstack = np.vstack((E_zstack, np.array([E_z])))
    H_xstack = np.vstack((H_xstack, np.array([nextH_x])))
    H_ystack = np.vstack((H_ystack, np.array([nextH_y])))
    J_zstack = np.vstack((J_zstack, np.array([nextJ_z])))
    
    # update variables 
    E_z = np.copy(nextE_z)
    H_x = np.copy(nextH_x)
    H_y = np.copy(nextH_y)
    J_z = np.copy(nextJ_z)
    
    
plt.rc('text', usetex=True)             # use LaTeX for text
plt.rc('font', family='serif')          # use serif font
plt.rcParams.update({'font.size': 15})  # set font size 

# Plotting 
plt.figure(figsize=(10,5))
if question =='c':
    plt.title('Question 2c, $\omega =$' + str(omega))
elif question =='e':
    plt.title('Question 2e, $\omega =$' +str(round(omega,2)))
plt.plot(t_array,H_xstack[:,16,0], alpha=0.8, linewidth=2,label='$H_x(t,0.5, 0)$')
plt.plot(t_array,H_ystack[:,0, 16],'--',label='$H_y(t,0,0.5)$' )
plt.plot(t_array,E_zstack[:,16,16], label='$E_z(t, 0.5,0.5)$')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.xlabel('time(s)')
plt.ylabel('Magnitude')
plt.tight_layout()
if question == 'c':
    plt.savefig('Q2c.pdf')
elif question =='e':
    plt.savefig('Q2e.pdf')
plt.show()



