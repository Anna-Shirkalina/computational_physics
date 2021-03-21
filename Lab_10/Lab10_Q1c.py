from numpy.random import random
import numpy as np
import matplotlib.pyplot as plt


L = 151  # number of positions on the square grid

x_init = (L - 1) // 2
y_init = (L - 1) // 2


def one_particle_trajectory(particles):
    """
    Return the trajectory of one particle until it stops moving
    Function has two parts, move right, left, down or up (randomly) then check
    if continued moves are still valid
    """
    x_path = [x_init]
    y_path = [y_init]
    can_move = True
    i = 0  # initialize the number of moves
    while can_move:
        r = random() # generate random number on the interval [0, 1)
        if 0 <= r < 0.25:  # move + x
            x_path.append(x_path[i] + 1)
            y_path.append(y_path[i])
        elif 0.25 <= r < 0.5:  # move - x
            x_path.append(x_path[i] - 1)
            y_path.append(y_path[i])
        elif 0.5 <= r < 0.75:  # move + y
            x_path.append(x_path[i])
            y_path.append(y_path[i] + 1)
        else:  # move - y
            x_path.append(x_path[i])
            y_path.append(y_path[i] - 1)
        # do not let particle escape from box. It can't move past L - 1 and 0
        if x_path[i + 1] >= L:
            x_path[i + 1] = L - 1
            can_move = False
        if y_path[i + 1] >= L:
            y_path[i + 1] = L - 1
            can_move = False
        if x_path[i + 1] <= 0:
            x_path[i + 1] = 0
            can_move = False
        if y_path[i + 1] <= 0:
            y_path[i + 1] = 0
            can_move = False
        if check_collision(particles, x_path[i + 1], y_path[i + 1]):
            # we want the particle stuck at the second last time step not on top
            # of the particle it collided with
            x_path[i + 1] = x_path[i]
            y_path[i + 1] = y_path[i]
            can_move = False
        i += 1  # number of moves
    return x_path, y_path


def check_collision(particles, x0, y0) -> bool:
    """check that a particle located at x0, y0 hasn't collided with any other
    particle
    """
    return particles[x0, y0] == 1


def check_center_square(particles, L) -> bool:
    """:return True if a particle is stuck at the center square, False otherwise
    """
    x0 = (L - 1) // 2
    y0 = (L - 1) // 2
    return particles[x0, y0] == 1


grid_of_stuck = np.zeros((L, L)) # 1 means particle is there, 0 means free space
center_occupied = False
i = 0  # index of the number of particles that have been generated
while not center_occupied:
    x_traj, y_traj = one_particle_trajectory(grid_of_stuck)
    grid_of_stuck[x_traj[-1], y_traj[-1]] = 1
    center_occupied = check_center_square(grid_of_stuck, L)
    i += 2


plt.figure(1)
plt.rc('font', family='serif')          # use serif font
plt.rcParams.update({'font.size': 13})  # set font size
plt.imshow(grid_of_stuck)
plt.title(f"DLA end positions for {i} particles")
plt.xlabel("Position on x-axis")
plt.ylabel("Position on y-axis")
plt.xlim((0, L))
plt.ylim((0, L))
plt.set_cmap("gray_r")
plt.show()
