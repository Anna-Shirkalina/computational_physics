import numpy as np
import matplotlib.pyplot as plt
from Lab06_Functions import *


N = 16
t_stop = 10
dt = 0.01
Lx = 4.0
Ly = 4.0

dx = Lx / np.sqrt(N)
dy = Ly / np.sqrt(N)

x_grid = np.arange(0.5 * dx, Lx, dx)
y_grid = np.arange(0.5 * dy, Ly, dy)



xx_grid, yy_grid = np.meshgrid(x_grid, y_grid)

x_initial = xx_grid.flatten()
y_initial = yy_grid.flatten()
position_init = np.stack((x_initial, y_initial), axis=1)

vel_init = np.zeros((N, 2))  # number of particles and 2 dimensions


time_array = np.arange(0, t_stop, dt)
num_steps = int(t_stop / dt)
velocity = np.zeros((num_steps, N, 2))
velocity_odd = np.zeros((num_steps, N, 2))
positions = np.zeros((num_steps, N, 2))
positions[0] = position_init
velocity_odd[0] = velocity[0] + dt * acceleration(positions[0])

for t in range(num_steps - 1):
    positions[t + 1] = positions[t] + dt * velocity_odd[t]
    k = dt * acceleration(positions[t + 1])
    velocity[t + 1] = velocity_odd[t] + 0.5 * k
    velocity_odd[t + 1] = velocity_odd[t] + k

plt.figure(1)
for particle in range(N):
    name = particle + 1
    plt.plot(positions[:, particle, 0], positions[:, particle, 1], label=name)

plt.plot(x_initial, y_initial, '.', label="Initial \n position of \n particles")
plt.xlabel("x distance")
plt.xlabel("y distance")
plt.legend(loc='right')
plt.xlim(right=13)


# calculate energy
energy = np.zeros(time_array.shape)
kinetic_energy = np.zeros(time_array.shape)
potential_energy = np.zeros(time_array.shape)

for i, t in enumerate(time_array):
    t = int(t)
    if i == 300:
        r = 10
    energy[i] = Kinetic(velocity[i]) + Potential(positions[i])
    kinetic_energy[i] = Kinetic(velocity[i])
    potential_energy[i] = Potential(positions[i])

plt.figure(2)

plt.plot(time_array, energy, label="total energy")
plt.plot(time_array, kinetic_energy, label="kinetic")
plt.plot(time_array, potential_energy, label="potential")
plt.xlabel("time")
plt.ylabel('Energy')
plt.legend(loc="best")
plt.show()
