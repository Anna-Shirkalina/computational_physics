import numpy as np
import scipy.special as fun
import matplotlib.pyplot as plt
from gaussxw import gaussxw
import numpy.linalg as la
import Lab4_Functions as func


def Hmn(m, n):
    """
    Return the nm element of the matrix produced by the Hamiltonian operator.
    Mass of the electron is 9.1094e-31kg, a=10eV, charge c= 1.6022e-19C, with
    with L = 5A
    :param m: the nth row of the matrix H
    :param n: the mth column of the matrix H
    :return: the nm element of the Hamiltonian as given in the physics
    description
    """
    a = 10 * 1.602176634e-19  # J
    h_bar = 1.054571817e-34  # J s .054571817×10−34
    M = 9.1094e-31  # kg
    L = 5e-10  # m
    # check that m =/= n and either both are even or both are odd
    if m != n and ((m % 2 != 0 and n % 2 != 0) or (m % 2 == 0 and n % 2 == 0)):
        return 0
    # check that m =/= n and one is odd and one is even
    if m != n and not (
            (m % 2 != 0 and n % 2 != 0) or (m % 2 == 0 and n % 2 == 0)):
        return (-8 * a * m * n) / ((np.pi ** 2) * ((m ** 2 - n ** 2) ** 2))
    # in this case m == n
    else:
        return 0.5 * a + ((np.pi ** 2) * (h_bar ** 2) * (m ** 2)) / (
                    2 * M * (L ** 2))


def H_matrix(N):
    """
    Return a N X N matrix H
    :param N: the size of the square matrix
    :return: a numpy array
    """
    H = np.zeros((N, N))
    for m in range(1, N + 1):
        for n in range(1, N + 1):
            H[m - 1, n - 1] = Hmn(m, n)
    return H


# use eigh to find eigenvectors and eigvalsh for eigenvalues
# (since H is symmetric)

# find eigenvalues for H of 10 X 10

eigenvectors = la.eigh(H_matrix(100))
eigenvalues_1 = la.eigvalsh(H_matrix(10))
eigenvalues_2 = la.eigvalsh(H_matrix(100))

print("ground energy in eV for N = 10, E =", eigenvalues_1[0] / 1.602176634e-19,
      "for N = 100, E = ", eigenvalues_2[0] * 6.242e18)


# question 2 e
def wave_function(phi_n, x, L):
    """Return the wave function at the point x given a well of length L, and an
    array of shrodinger coefficients
    phi_n : 1D numpy array
    x: 1D array
    """
    # make an array of n's from 1 to the length of phi_n
    n = np.arange(1, phi_n.size + 1)
    # make an array of the sin term within the sum
    sin = np.sin(n * x * np.pi / L)
    # multiply element wise and add
    return np.sum(np.multiply(sin, phi_n))


def wave_function_density(phi_n, x, L):
    """Return the wave function density squared"""
    return np.abs(wave_function(phi_n, x, L)) ** 2


eigenvalues, eigenvectors = la.eigh(H_matrix(100))

# pull out the fourier coefficients
phi_ground = eigenvectors[:, 0]
phi_first_excited = eigenvectors[:, 1]
phi_second_excited = eigenvectors[:, 2]

L = 5e-10  # m
# integrate the wave function density
A_ground = func.gauss(0, L, 500, lambda z: wave_function_density(phi_ground, z, L))
A_first = func.gauss(0, L, 500, lambda z: wave_function_density(phi_first_excited, z, L))
A_second = func.gauss(0, L, 500, lambda z: wave_function_density(phi_second_excited, z, L))


# Make sure the probability density is normalized
A_ground_i = func.gauss(0, L, 500, lambda z: wave_function_density(phi_ground / np.sqrt(A_ground), z, L))
A_first_i = func.gauss(0, L, 500, lambda z: wave_function_density(phi_first_excited /np.sqrt(A_first), z, L))
A_second_i = func.gauss(0, L, 500, lambda z: wave_function_density(phi_second_excited/ np.sqrt(A_second), z, L))


print("Normalized intergral for the probability density in the ground state", A_ground_i)
print("Normalized intergral for the probability density in the first excited state", A_first_i)
print("Normalized intergral for the probability density in the second excited state", A_second_i)



x_array = np.linspace(0, L, 100)

plt.figure(1)

plt.rcParams.update({'font.size': 11})  # set font size
wave_ground = np.zeros(x_array.shape)
wave_first = np.zeros(x_array.shape)
wave_second = np.zeros(x_array.shape)
for i, x in enumerate(x_array):
    wave_ground[i] = wave_function_density(phi_ground / np.sqrt(A_ground), x, L)
    wave_first[i] = wave_function_density(phi_first_excited / np.sqrt(A_first), x, L)
    wave_second[i] = wave_function_density(phi_second_excited / np.sqrt(A_second), x, L)
plt.plot(x_array, wave_ground, label="Ground State")
plt.plot(x_array, wave_first, label="First Excited State")
plt.plot(x_array, wave_second, label="Second Excited State")
plt.xlabel("Distance within the quantum well, in m      ")
plt.ylabel(rf"Probability density, $|\psi (x)|^2$")
plt.legend(loc='upper left')
plt.show()

