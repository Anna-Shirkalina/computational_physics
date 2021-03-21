"""
Lab6 PHY407
@author: Genevieve Beauregard(primarily), Anna Shirkalina
Q2a
"""
import numpy as np
import matplotlib.pyplot as plt
from Lab06_Functions import *


sigma = 1
epsilon = 1
m = 1

r1_initial =[2,3]
r2_initial =[3.5, 4.4]

# Initialize constants
r1 = np.array(r1_initial, dtype=float)
r2 = np.array(r2_initial, dtype=float)

v1 = np.array([0,0], dtype=float)
v2 =np.array([0,0], dtype=float)

# values to be updated
v = np.array([v1,v2], dtype=float)

r = np.array([r1,r2], dtype=float)


# Initialize number of steps
N = 100
dt = 0.01
time = np.arange(0, dt*N, dt)
h = dt



# Def f(r):
    # return np.array([[a1x, a1y], [a2x, a3y]], dtype=float), as per our found
    # equations

def f(r):
    """This is as defined in the equation handout. This takes in a vector
    input r =[r1, r2] where r1 and r2 are the vectorial positions of particle
    1 and 2 respectively. This returns [a1, a2] for the associated timestep,
    where a1 and a2 are similarly are the associated acceleration vectors.
    Input:
        r = np.array([r1, r2]) = np.array([[x1, y1],
                                           [x2,y2]])
        with float entries

    Output:
        a = np.array([a1, a2]) = np.array([[ax1, ay1],
                                           [ax2, ay2]])
        with float entries
    """

    # obtain constants
    x1 = r[0,0]
    x2 = r[1,0]
    y1 = r[0,1]
    y2 = r[1,1]


    # intoducing some new terms to simplify. Refer to lab report for equations
    # in 2a)
    x = x2 - x1
    y = y2 - y1

    ax1 = 12* (- 2*(x)/(x**2 + y**2)**7 + (x)/(x**2 + y**2)**4)
    ax2 = -ax1 # newton's third law

    ay1 = 12 *( - 2*(y)/(x**2 + y**2)**7 + (y)/(x**2 + y**2)**4)
    ay2 = -ay1

    return np.array([[ax1, ay1], [ax2, ay2]])


# # define the following helper functions for the verlet loop
# # Eqn corresponds to the equation number in the lab handout

def Eqn8(r, v, h):
    """ Returns next step of vector input r for vector input r, and vector
    input v, with time step h. Refer to Eqn 8 in Lab Handout
    Input:
        Current r = np.array([r1, r2]) = np.array([[x1, y1],
                                                   [x2,y2]])
        current v = np.array([v1, v2]) = np.array([[vx1, vy1],
                                                     [vx2, vy2]])
        h, scalar time step
    Output:
        Updated r array
    """
    return r + h * v


def Eqn9(r, h):
    """Returns k array
    Input:
        updated r array  r =[r1, r2]
        h, scalar time step
    Output:
        k =np.array([k1, k2]) = np.array([[kx1, ky1],
                                          [kx2, ky2]])
    """
    return h * f(r)


def Eqn10(v, k):
    """Returns v array associated with same time as updated r -array.
    Not necessary for plotting, but needed for energy plots.
    Input:
        current v = np.array([v1, v2])
        k = np.array([k1, k2])
    Ouput:
        v associated with updated r
    """

    return v + 0.5 * k


def Eqn11(v, k):
    """Returns updated v array.
    Input:
        prev v array, np.array([v1, v2])
        k = np.array([k1, k2])

    Output:
        updated v array
    """

    return v + k


# Obtain first step of v using equation 7 of handout
v = v + 0.5 * h * f(r)

# Initialize accumalator variables, values to appended
# use list due to easier time appending
r1_array = [r1.tolist()]
r2_array =[r2.tolist()]

v1_array = [v[0].tolist()]
v2_array = [v[1].tolist()]

# for energy stuff
v1_energy = [v1.tolist()]
v2_energy = [v2.tolist()]
v_energy = np.array([v1, v2])



# loop over variables
for i in range(1, N):
    # update r array, r(t + h) array with equation 8
    r = Eqn8(r, v, h)

    #collect values
    r1_array.append(r[0].tolist())
    r2_array.append(r[1].tolist())

    # obtain k vector with equation 9
    k = Eqn9(r, h)

    # obtain v(t + h) with equation 10
    v_energy = Eqn10(v, k)

    # obtain next step for v array, v(t + 3/2 h) with equation 11
    v = Eqn11(v, k)

    # collect values
    v1_array.append(v[0].tolist())
    v2_array.append(v[1].tolist())


# Obtain r_1, r_2 and plot trajectories

# converted back to array as obtaining columns are a bit easier with np array
r1_array = np.array(r1_array)
r2_array = np.array(r2_array)
v1_array = np.array(v1_array)
v2_array = np.array(v2_array)
positions = np.stack((r1_array, r2_array), axis=1)
velocities = np.stack((v1_array, v2_array), axis=1)

# calculate energy for the particle motion
num_time_steps = time.shape[0]
energy = np.zeros((num_time_steps))
kinetic_energy = np.zeros(num_time_steps)
potential_energy = np.zeros(num_time_steps)

for i, t in enumerate(time):
    t = int(t)
    energy[i] = Kinetic(velocities[i]) + Potential(positions[i])
    kinetic_energy[i] = Kinetic(velocities[i])
    potential_energy[i] = Potential(positions[i])

plt.figure(2)
plt.title('Energy against time. Initial conditons, $r_1 =${}, $r_2 =$ {}'\
          .format(str(r1_initial),str(r2_initial)))
plt.plot(time, energy, label="total energy")
plt.plot(time, kinetic_energy, label="kinetic")
plt.plot(time, potential_energy, label="potential")
plt.xlabel('Time')
plt.ylabel('Energy')
plt.legend(loc="best")
plt.savefig('Q2c_energyiii.pdf')


plt.figure(figsize=(5,5))
plt.title('Initial conditons trajectory, $r_1 =${}, $r_2 =$ {}'.\
          format(str(r1_initial),str(r2_initial)))
plt.plot(r1_array[:,0], r1_array[:,1], '.', label='Particle 1')
plt.plot(r2_array[:,0], r2_array[:,1], '.', label='Particle 2')
plt.axis('equal')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.savefig('Q2c_iii.pdf')

#plt.savefig('Q2bii.pdf')




# for oscillatory motion checks
plt.figure()
plt.title("x against time, $r_1 =${}, $r_2 =$ {}".format(str(r1_initial),\
                                                         str(r2_initial)))
plt.plot(time, r1_array[:,0], label='Particle 1')
plt.plot(time, r2_array[:,0], label='Particle 2')
plt.legend()
plt.ylabel('x')
plt.xlabel('Time')
plt.savefig('Q2c_xtimeiii.pdf')


# for oscillatory motion checks
plt.figure()
plt.title("y against time, $r_1 =${}, $r_2 =$ {}".format(str(r1_initial),\
                                                         str(r2_initial)))
plt.plot(time, r1_array[:,1], label='Particle 1')
plt.plot(time, r2_array[:,1], label='Particle 2')
plt.legend()
plt.ylabel('y')
plt.xlabel('Time')
plt.show()
plt.savefig('Q2c_ytimeiii.pdf')
