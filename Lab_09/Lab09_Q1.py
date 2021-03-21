import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as pc
# import pickle  # package used to save array as binary file
from tqdm import tqdm
h_bar = pc.hbar
m = pc.m_e  # electron mass


def v_square(x):
    """Return the potential for the electron inside the boundaries of
    -L/2 to L/2 for a square well"""
    return 0 * x

def v_harmonic(x, omega):
    """Return the potential for the electron inside the boundaries of
        -L/2 to L/2 for a harmonic oscillator"""
    return 0.5 * m * (omega**2) * (x**2)

def v_double_well(x, v_0, x_1):
    """Return the potential for the electron inside the boundaries of
        -L/2 to L/2 for a double well"""
    return v_0 * (((x**2 / x_1 ** 2) - 1) ** 2)

def psi(x, t, x0, sigma, kappa):
    """:return psi(x, t)"""
    if t == 0:
        exponent = -((x - x0)**2) / (4 * (sigma**2)) + 1j * kappa * x
        return 1 * np.exp(exponent)
    return 0

def build_psi(p, a, t, x0, sigma, kappa):
    """Return the psi array"""
    L = a * p
    psi_array = np.zeros(p - 1, complex)
    for i in range(1, p):
        psi_array[i - 1] = psi(i * a - L / 2, t, x0, sigma, kappa)
    return psi_array

def build_hamiltonian(p, a, V):
    """Return the discretized hamiltonian"""
    A = - h_bar**2 / (2 * m * (a**2)) * np.ones(p - 2)
    B = V(np.arange(1, p) * a - 0.5 * a * p) + 2 * h_bar**2 / (2 * m * (a**2))
    H = np.diag(A, k=-1) + np.diag(B, k=0) + np.diag(A, k=1)
    return H

def energy(psi, hamiltonian, h):
    """Return the energy at time t, as given in equation 4 in the lab-manual
    a and b are the boundaries of integration and h is the number of integration
    steps, technically the integration happens over infinity, but psi is zero
    outside of the [a, b] domain
    """
    psi_conj = np.conj(psi)
    array = np.multiply(psi_conj, np.matmul(hamiltonian, psi))
    e = trap(array, h)
    return e

def trap(f_array, h):
    """Numerically computes the integral using the trapezoidal method of an
    input function func of a single variable, from a to b with N slices.
    Based on Eqn 5.3 of Newman."""

    # Calculate integral using trap method based on Eqn 5.3 of the textbook

    # The end bits
    s = 0.5*f_array[0] + 0.5*f_array[-1]
    sum = np.sum(f_array[1:-1]) + s

    I_trap = sum * h
    return I_trap

def normalize_psi(psi, h):
    """Find the normalization constant for psi s.t.
     integral from -inf to inf phi* psi dx = 1, then divide psi by that constant
     """
    psi_conj = np.conj(psi)
    array = np.multiply(psi_conj, psi)
    norm_constant = np.sqrt(np.real(trap(array, h)))
    return psi / norm_constant, norm_constant

def expectation_x(psi, a, b, h, p):
    """Return the expectation value, <X> at time t
    a and b are the boundaries of integration and h is the number of integration
    steps, technically the integration happens over infinity, but psi is zero
    outside of the [a, b] domain
    """
    psi_conj = np.conj(psi)
    x_array = np.linspace(a, b, p - 1)
    array = np.multiply(psi_conj, np.multiply(x_array, psi))
    return trap(array, h)

# define constants
L = 10 ** (-8)  # m
p = 1024
tau = 10 ** (-18)  # s
a = L / p
N = 3000  # number of integration time steps
sigma = L / 25
kappa = 500 / L
x0 = L / 5
omega = 3e15

# initialize Hamiltonian and Psi_initial
H_d = build_hamiltonian(p, a, lambda x: v_square(x))
Psi_initial = build_psi(p, a, 0, x0, sigma, kappa)
Psi_initial, Psi_norm_init = normalize_psi(Psi_initial, a)

# initialize arrays for storage of Psi through time
Psi_n = np.zeros((N, Psi_initial.size), complex)
Psi_n[0, :] = Psi_initial

# Define the L and R matrices
L_matrix = np.eye(p - 1) + 0.5j * (tau / h_bar) * H_d
R_matrix = np.eye(p - 1) - 0.5j * (tau / h_bar) * H_d

# run the Crank-Nicolson method of integration
for t in tqdm(range(N - 1)):
    v_vector = np.matmul(R_matrix, Psi_n[t])
    Psi_n[t + 1] = np.linalg.solve(L_matrix, v_vector)

# save the psi array as a binary file
#pickle.dump(Psi_n, open("./Psi_n.pickle", "wb"))

# Normalization, energy and the trajectories
Psi_norm = np.zeros(N, complex)
Psi_expected = np.zeros(N, complex)
Energy = np.zeros(N, complex)
for t in range(N):
    Psi_normed, Psi_norm[t] = normalize_psi(Psi_n[t], a)
    Psi_expected[t] = expectation_x(Psi_n[t], -L / 2, L / 2, a, p)
    Energy[t] = energy(Psi_n[t + 1], H_d, a)



# graph the energy
plt.figure(1)
time = np.arange(0, N)
plt.plot(time, np.real(Energy))
plt.xlabel(f"Number of integration time steps $\\tau$ = {tau} s")
plt.ylabel(f"Energy in the system in Joules")
plt.title("The energy of an electron in a square well potential")

# graph the normalization factor
plt.figure(2)
time = np.arange(0, N - 1)
plt.plot(time, np.real(np.sqrt(Psi_norm[0:-1])))
plt.xlabel(f"Number of integration time steps $\\tau$ = {tau} s")
plt.ylabel(f"The normalization factor for $\Psi$")
plt.title("The normalization factor for $\Psi$ during the integration")

# graph the probability density
plt.figure(3)
x_array = np.linspace(-L/4, L/4, p - 1)
plt.subplot(2, 2, 1)
plt.plot(x_array, np.real(np.multiply(Psi_n[0], np.conj(Psi_n[0]))))
plt.ylabel(f"Probability density function")
plt.title(f"1st integration step")
plt.subplot(2, 2, 2)
plt.plot(x_array, np.real(np.multiply(Psi_n[int(N/4)], np.conj(Psi_n[int(N/4)]))))
plt.title(f"{int(N/4)} integration steps")
plt.subplot(2, 2, 3)
plt.plot(x_array, np.real(np.multiply(Psi_n[int(N/2)], np.conj(Psi_n[int(N/2)]))))
plt.xlabel(f"The location in the square well in m")
plt.ylabel(f"Probability density function")
plt.title(f"{int(N/2)} integration steps")
plt.subplot(2, 2, 4)
plt.plot(x_array, np.real(np.multiply(Psi_n[int(N - 1)], np.conj(Psi_n[int(N -1)]))))
plt.xlabel(f"The location in the square well in m")
plt.title(f"{int(N)} integration steps")
plt.suptitle(f"The probability density functions for 3000 time steps with $\\tau$={tau}")

# graph the trajectory
plt.figure(4)
time = np.arange(0, N)
plt.plot(time, Psi_expected)
plt.xlabel(f"Number of integration time steps $\\tau$ = {tau} s")
plt.ylabel(f"The trajectory, $<\Psi>$")
plt.title("The trajectory, $<\Psi>$ during the integration")

plt.show()
