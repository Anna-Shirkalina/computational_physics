import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as pc
from Lab07_Functions import simpsons

a = pc.physical_constants['Bohr radius'][0]

E0 = pc.physical_constants['Rydberg constant times hc in eV'][0]
e = pc.e
h_bar = pc.hbar
m = pc.m_e  # electron mass
epsilon_0 = pc.epsilon_0

def V(r):
    """Return the potential experienced by a an electron interacting
    electrostatically with a proton"""
    return -(e ** 2) / (4 * np.pi * epsilon_0 * r)


def f(r, x, E, l):
    """Return the systems of equations for the schordinger functions
    Based on the 2nd order ODE as given in the lab mannual for equation 2"""
    R = r[0]
    S = r[1]

    fR = S
    fS = ((2*m/h_bar**2)*(V(x)-E)*R * (x ** 2) + (l + 1) * l * R - 2 * x * S) / (x ** 2)
    #(l*(l+1)*R + (2*m_e*r**2 / hbar**2) * (V(r)-E) * R - 2*r*S) / r**2
    return np.array([fR, fS], float)


# Calculate the wavefunction for a particular energy
def solve(E, x0, x_L, h, l):
    """The rutta-kunga method for the two coupled first order ODE's
    for solving the time -independent schrodinger equation in search of the
    energy"""
    R = 0.0
    S = 1.0
    r = np.array([R, S], float)
    for x in np.arange(x0, x_L, h):
        k1 = h*f(r, x, E, l)
        k2 = h*f(r+0.5*k1, x+0.5*h, E, l)
        k3 = h*f(r+0.5*k2, x+0.5*h, E, l)
        k4 = h*f(r+k3, x+h, E, l)
        r += (k1+2*k2+2*k3+k4)/6
    return r[0]


# Main program to find the energy using the secant method



def state_energy(n, l, h, r_left, r_inf):
    """return the energy cooresponding to the state of the
    hydrogen atom using the secant method"""
    E1 = -15 * e / (n ** 2)
    E2 = -13 * e / (n ** 2)
    psi2 = solve(E1, r_left, r_inf, h, l)
    target = e/1000
    while abs(E1-E2) > target:
        psi1, psi2 = psi2, solve(E2, r_left, r_inf, h, l)
        E1, E2 = E2, E2-psi2*(E2-E1)/(psi2-psi1)
    return E2/e


def runge_kutta(E, x0, x_L, h, l):
    """Return rutta-kunga method for the two coupled first order ODE's
    for solving the time -independent schrodinger equation"""
    R = 0.0
    S = 1.0
    r = np.zeros((int((x_L - x0) / h) + 1, 2))
    r[0] = [R, S]
    for i, x in enumerate(np.arange(x0, x_L, h)):
        k1 = h*f(r[i], x, E, l)
        k2 = h*f(r[i] + 0.5 * k1, x + 0.5 * h, E, l)
        k3 = h*f(r[i] + 0.5 * k2, x + 0.5 * h, E, l)
        k4 = h*f(r[i] + k3, x + h, E, l)
        r[i + 1] = r[i] + (k1+2*k2+2*k3+k4)/6
    return r[:-1, :]


def theory_solution(r, n, l):
    """The explicit solution for the hydrogen wave function n = 0, l = 0
    as found on Hyperphysics
    2/ a_0^{3/2} * exp(-r / a_0), where a_0 = h_bar / m * e^2
    """
    a_0 = (h_bar ** 2) / (m * (e ** 2)) * 4 * np.pi * pc.epsilon_0
    a_32 = a_0 ** (3 / 2)
    e_1 = np.exp(-r / a_0)
    e_2 = np.exp(-r / (2 * a_0))
    if n == 1 and l == 0:
        return (2 / a_32) * e_1
    if n == 2 and l == 0:
        return (1 / (2 * np.sqrt(2) * a_32)) * (2 - r / a_0) * e_2
    if n == 2 and l == 1:
        return (1 / (2 * np.sqrt(6) * a_32)) * (r / a_0) * e_2
    else:
        return None



n = 2
l = 0

r_inf = 200 * a  # intialize the boundary location where E is close to zero
h = 0.001 * a  # define step size
r_left = 0.001 * a
r_right = 15 * a # initialize the boundary for graphing purposes
r_right_index = int((r_right - r_left) / h) + 1

# calculate the energies
E_0 = state_energy(1, 0, h, r_left, r_inf)
E_1 = state_energy(2, 0, h, r_left, r_inf)
E_3 = state_energy(2, 1, h, r_left, r_inf)

print(f"E_0 = , {E_0}")
print(f"E_1 = , {E_1}")
print(f"E_3 = , {E_3}")

x_array = np.arange(r_left, r_right, h) # initialize the x axis array
r_0 = runge_kutta(E_0, r_left, r_inf, h, 0) # calculate the wave function
r_1 = runge_kutta(E_1, r_left, r_inf, h, 0)
r_3 = runge_kutta(E_3, r_left, r_inf, h, 1)
# normalize the wave function
r_0_norm = simpsons(r_left, r_inf, np.square(r_0[:, 0]), h)
r_1_norm = simpsons(r_left, r_inf, np.square(r_1[:, 0]), h)
r_3_norm = simpsons(r_left, r_inf, np.square(r_3[:, 0]), h)

r_0 = r_0[:, 0] / np.sqrt(r_0_norm)
r_1 = r_1[:, 0] / np.sqrt(r_1_norm)
r_3 = r_3[:, 0] / np.sqrt(r_3_norm)

# find the analytical solution
theory_r0 = np.zeros(x_array.shape)
theory_r1 = np.zeros(x_array.shape)
theory_r2 = np.zeros(x_array.shape)

for i, r in enumerate(x_array):
    theory_r0[i] = theory_solution(r, 1, 0)
    theory_r1[i] = theory_solution(r, 2, 0)
    theory_r2[i] = theory_solution(r, 2, 1)

# normalize the analytical solution
theory_0_norm = simpsons(r_left, r_inf, np.square(theory_r0), h)
theory_1_norm = simpsons(r_left, r_inf, np.square(theory_r1), h)
theory_3_norm = simpsons(r_left, r_inf, np.square(theory_r2), h)

theory_r0 = theory_r0 / np.sqrt(theory_0_norm)
theory_r1 = theory_r1 / np.sqrt(theory_1_norm)
theory_r2 = theory_r2 / np.sqrt(theory_3_norm)



# plot
plt.figure(1)
plt.rc('font', family='serif')          # use serif font
plt.rcParams.update({'font.size': 13})
plt.plot(x_array / a, r_0[:r_right_index], label="Computational solution, ground state \n n = 1, l = 0")
plt.plot(x_array / a, r_1[:r_right_index], label="Computationa solution, first excited state \n n = 2, l = 0")
plt.plot(x_array / a, r_3[:r_right_index], label="Compuational solution, first excited state \n n = 2, l = 1")
plt.plot(x_array / a, theory_r0, label="Analytical solution, ground state \n n = 1, l = 0")
plt.plot(x_array / a, theory_r1, label="Analytical solution, first excited state \n n = 2, l = 0")
plt.plot(x_array / a, theory_r2, label="Analytical solution, first excited state \n n = 2, l = 1")
plt.legend(loc='best')
plt.xlabel("Distance from center of the atom, in Bohr radius units")
plt.ylabel(r'Normalized Wave function')
plt.show()
