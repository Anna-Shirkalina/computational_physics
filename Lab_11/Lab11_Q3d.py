"""
Strter code for protein folding
Author: Nicolas Grisuard, based on a script by Paul Kushner
"""

from random import random, randrange
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from tqdm import tqdm
import pickle as pickle


def calc_energy(monomer_coords, monomer_array):
    """ Compute energy of tertiary structure of protein """
    energy = 0.0

    # compute energy due to all adjacencies (incl. directly bonded monomers)
    for i in range(N):
        for nghbr in [[-1, 0], [1, 0], [0, -1], [0, 1]]:  # 4 neighbours
            nghbr_monomer = monomer_array[monomer_coords[i, 0] + nghbr[0],
                                          monomer_coords[i, 1]+nghbr[1]]

            if nghbr_monomer == 1:  # check neighbour is not empty
                energy += eps

    # divide by 2 to correct for double-counting
    energy = .5*energy

    # correct energy to not count directly bonded monomer neighbours
    energy -= (N-1)*eps

    return energy


def dist(position1, position2):
    """ Compute distance """
    return ((position1[0]-position2[0])**2+(position1[1]-position2[1])**2)**.5


font = {'family': 'DejaVu Sans', 'size': 14}  # adjust fonts
rc('font', **font)
dpi = 150


eps = -5.0  # interaction energy
N = 30  # length of protein
T_f = 0.5  # temperature for Monte Carlo
dT = 0.5
T_i = 10
T_steps = int((T_i - T_f) / dT)
num_temp_steps = int(5e5)
n = int(num_temp_steps * T_steps)  # number of Monte Carlo steps
T_array = np.arange(T_i, T_f, -dT)


energy_array = np.zeros(num_temp_steps)  # initialize array to hold energy
energy_temp_array = np.zeros((T_steps, 2))

# initialize arrays to store protein information
# 1st column is x coordinates, 2nd column is y coordinates, of all N monomers
monomer_coords = np.zeros((N, 2), dtype='int')

# initialize position of polymer as horizontal line in middle of domain
monomer_coords[:, 0] = range(N//2, 3*N//2)
monomer_coords[:, 1] = N

# 2D array representing lattice,
# equal to 0 when a lattice point is empty,
# and equal to 1 when there is a monomer at the lattice point
monomer_array = np.zeros((2*N+1, 2*N+1), dtype='int')

# fill lattice array
for i in tqdm(range(N)):
    monomer_array[monomer_coords[i, 0], monomer_coords[i, 1]] = 1

# calculate energy of initial protein structure
energy = calc_energy(monomer_coords, monomer_array)

# do Monte Carlo procedure to find optimal protein structure
for step in tqdm(range(T_steps)):
    for j in range(int(num_temp_steps)):
        energy_array[j] = energy

        # move protein back to centre of array
        shift_x = int(np.mean(monomer_coords[:, 0])-N)
        shift_y = int(np.mean(monomer_coords[:, 1])-N)
        monomer_coords[:, 0] -= shift_x
        monomer_coords[:, 1] -= shift_y
        monomer_array = np.roll(monomer_array, -shift_x, axis=0)
        monomer_array = np.roll(monomer_array, -shift_y, axis=1)

        # pick random monomer
        i = randrange(N)
        cur_monomer_pos = monomer_coords[i, :]

        # pick random diagonal neighbour for monomer
        direction = randrange(4)

        if direction == 0:
            neighbour = np.array([-1, -1])  # left/down
        elif direction == 1:
            neighbour = np.array([-1, 1])  # left/up
        elif direction == 2:
            neighbour = np.array([1, 1])  # right/up
        elif direction == 3:
            neighbour = np.array([1, -1])  # right/down

        new_monomer_pos = cur_monomer_pos + neighbour

        # check if neighbour lattice point is empty
        if monomer_array[new_monomer_pos[0], new_monomer_pos[1]] == 0:
            # check if it is possible to move monomer to new position without
            # stretching chain
            distance_okay = False
            if i == 0:
                if dist(new_monomer_pos, monomer_coords[i+1, :]) < 1.1:
                    distance_okay = True
            elif i == N-1:
                if dist(new_monomer_pos, monomer_coords[i-1, :]) < 1.1:
                    distance_okay = True
            else:
                if dist(new_monomer_pos, monomer_coords[i-1, :]) < 1.1 \
                   and dist(new_monomer_pos, monomer_coords[i+1, :]) < 1.1:
                    distance_okay = True

            if distance_okay:
                # calculate new energy
                new_monomer_coords = np.copy(monomer_coords)
                new_monomer_coords[i, :] = new_monomer_pos

                new_monomer_array = np.copy(monomer_array)
                new_monomer_array[cur_monomer_pos[0], cur_monomer_pos[1]] = 0
                new_monomer_array[new_monomer_pos[0], new_monomer_pos[1]] = 1

                new_energy = calc_energy(new_monomer_coords, new_monomer_array)

                if random() < np.exp(-(new_energy-energy)/T_array[step]):
                    # make switch
                    energy = new_energy
                    monomer_coords = np.copy(new_monomer_coords)
                    monomer_array = np.copy(new_monomer_array)
    # save the energy at the end of the iteration
    energy_T = np.copy(energy_array)
    energy_temp_array[step] = np.array([np.mean(energy_T), np.std(energy_T)])
    energy_array = np.zeros(energy_array.size)

pickle.dump(energy_temp_array, open("q3d_energy_per_temp_array.pickle", "wb"))


plt.figure()
plt.title('$T$ = {0:.1f}, $N$ = {1:d}'.format(T_f, N))
plt.plot(energy_array)
plt.xlabel('MC step')
plt.ylabel('Energy')
plt.grid()
plt.tight_layout()
plt.savefig('energy_vs_step_T{0:d}_N{1:d}_n{2:d}.pdf'.format(int(10*T_f), N, n),
            dpi=dpi)

fig, axes = plt.subplots()
plt.errorbar(T_array, energy_array[:, 0], yerr=energy_array[:, 1])
plt.xlabel('Temperature')
plt.ylabel('Energy')
plt.title("Energy dependence on temperature")
axes.invert_xaxis()
plt.show()

