import numpy as np


def Kinetic(velocity):
    """Calculate the total kinetic energy given an array of velocities of
    particles"""
    n = int(velocity.shape[0])
    energy = 0
    for i in range(n):
        v = velocity[i][0] ** 2 + velocity[i][1] ** 2
        energy += v
    return energy * 0.5


def Potential(position):
    """Calculate the total potential energy given an array position of particles
    """
    n = int(position.shape[0])
    energy = 0
    for i in range(n):
        for j in range(n):
            r_x = position[j, 0] - position[i, 0]
            r_y = position[j, 1] - position[i, 1]
            r = np.sqrt(r_x ** 2 + r_y ** 2)
            if r != 0:
                energy += 4 * (1 / (r ** 12) - 1 / (r ** 6))

    #  we need to scale the potential by 0.5 (bc the potential is 0.5 particle)
    #  and by another 0.5 because of our double for loop which double counts
    #  the energy
    return energy * 0.25


def acceleration_helper(x, y):
    """Helper function for the main acceleration function, returns the
    acceleration for a given r = (x, y) between particles"""
    denominator = (x ** 2 + y ** 2)
    if denominator == 0:
        return 0, 0
    a_x = 12 * x * ((-2 / (denominator ** 7)) + (1 / (denominator ** 4)))
    a_y = 12 * y * (((-2) / (denominator ** 7)) + (1 / (denominator ** 4)))
    return a_x, a_y


def acceleration(position_matrix):
    """Return the acceleration matrix at time t, given the position matrix at
    time t. The position matrix is of the form (i, 2) where
    i = num_particles.
    Return a matrix of size (i, 2)
    """

    n = position_matrix.shape[0]
    acceleration_array = np.zeros((n, 2))

    for i in range(n):
        a_particle_i = np.zeros((n, 2))
        for j in range(n):
            r_x = position_matrix[j, 0] - position_matrix[i, 0]
            r_y = position_matrix[j, 1] - position_matrix[i, 1]
            a_x, a_y = acceleration_helper(r_x, r_y)
            a_particle_i[j] = np.array((a_x, a_y))
        acceleration_array[i] = np.sum(a_particle_i, axis=0)

    return acceleration_array


def build_block(center_tile, length_x, length_y):
    """build up 9 tiles surrounding the main block that contains N particles
    to a total of N * 9 particles
    """

    N = center_tile.shape[0]
    tile5 = np.copy(center_tile)

    tile1 = np.copy(center_tile) + np.ones((N, 2)) * np.array([-length_x, length_y])
    tile2 = np.copy(center_tile) + np.ones((N, 2)) * np.array([0, length_y])
    tile3 = np.copy(center_tile) + np.ones((N, 2)) * np.array([length_x, length_y])
    tile4 = np.copy(center_tile) + np.ones((N, 2)) * np.array([-length_x, 0])
    tile6 = np.copy(center_tile) + np.ones((N, 2)) * np.array([length_x, 0])
    tile7 = np.copy(center_tile) + np.ones((N, 2)) * np.array([-length_x, -length_y])
    tile8 = np.copy(center_tile) + np.ones((N, 2)) * np.array([0, -length_y])
    tile9 = np.copy(center_tile) + np.ones((N, 2)) * np.array([length_x, -length_y])
    big_block = np.vstack(
        (tile1, tile2, tile3, tile4, tile5, tile6, tile7, tile8, tile9))
    if not np.array_equal(big_block[N * 4: N * 5], tile5):
        print(-1)
    return big_block
