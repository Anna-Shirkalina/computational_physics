import numpy as np
import matplotlib.pyplot as plt
import random as random
from tqdm import tqdm


N = 25
R = 0.02
Tmax = 10.0
Tmin = 1e-3
tau = 1e2


t = 0
T = Tmax
num_iter = 20
second_seeds = range(10, num_iter + 10)
Distances = [None] * num_iter


# Function to calculate the magnitude of a vector
def mag(x):
    return np.sqrt(x[0]**2+x[1]**2)


# Function to calculate the total length of the tour
def distance(r, N):
    s = 0.0
    for i in range(N):
        s += mag(r[i+1]-r[i])
    return s


# Choose N city locations and calculate the initial distance
r = np.zeros([N+1, 2], float)
random.seed(10)
for i in range(N):
    r[i, 0] = random.random()
    r[i, 1] = random.random()
r[N] = r[0]
D = distance(r, N)

plt.figure(1)
plt.plot(r[:, 0], r[:, 1], '-k')  # plot bonds
#plt.title('$T$ = {0:.1f}, Energy = {1:.1f}'.format(T_f, energy))
# plot monomers
for i in range(N):
    plt.plot(r[i, 0], r[i, 1], '.r', markersize=15)
plt.tight_layout()
plt.title("Original path layout")


# Main loop
for iteration in tqdm(range(num_iter)):
    curr_d = []
    T = Tmax
    t = 0

    # Choose N city locations and calculate the initial distance
    r = np.zeros([N + 1, 2], float)
    random.seed(10)
    for i in range(N):
        r[i, 0] = random.random()
        r[i, 1] = random.random()
    r[N] = r[0]
    a = np.copy(r[:, :])
    D = distance(r, N)
    random.seed(second_seeds[iteration])
    while T > Tmin:

        # Cooling
        t += 1
        T = Tmax*np.exp(-t/tau)

        # Choose two cities to swap and make sure they are distinct
        i, j = random.randrange(1, N), random.randrange(1, N)
        while i == j:
            i, j = random.randrange(1, N), random.randrange(1, N)

        # Swap them and calculate the change in distance
        oldD = D
        r[i, 0], r[j, 0] = r[j, 0], r[i, 0]
        r[i, 1], r[j, 1] = r[j, 1], r[i, 1]
        D = distance(r, N)
        deltaD = D - oldD

        # If the move is rejected, swap them back again
        if random.random() > np.exp(-deltaD/T):
            r[i, 0], r[j, 0] = r[j, 0], r[i, 0]
            r[i, 1], r[j, 1] = r[j, 1], r[i, 1]
            D = oldD

        curr_d.append(D)
    Distances[iteration] = curr_d[-1]
    if iteration == 3:
        path3 = np.copy(r[:, :])
    if iteration == 7:
        path7 = np.copy(r[:, :])


plt.figure(2)
plt.plot(range(len(Distances)), Distances)
plt.xlabel("Iteration number")
plt.ylabel("Total Distance")
plt.title(f"Variance of Distance for {len(Distances): .2f} iteration, mean = "
          f"{np.mean(Distances): .2f} with a standard deviation = {np.std(Distances): .2f}")


plt.figure(3)
plt.plot(path3[:, 0], path3[:, 1], '-k')  # plot bonds
# plot monomers
for i in range(N):
    plt.plot(path3[i, 0], path3[i, 1], '.r', markersize=15)
plt.tight_layout()
plt.title(f"Final layout for the third path, tau={tau} ")


plt.figure(4)
plt.plot(path7[:, 0], path7[:, 1], '-k')  # plot bonds
# plot monomers
for i in range(N):
    plt.plot(path7[i, 0], path7[i, 1], '.r', markersize=15)
plt.tight_layout()
plt.title(f"Final layout for the seventh path, tau={tau}")


plt.figure(6)
plt.plot(path7[:, 0], path7[:, 1], '-k')  # plot bonds
# plot monomers
for i in range(N):
    plt.plot(path7[i, 0], path7[i, 1], '.r', markersize=15)
plt.tight_layout()
plt.title(f"Final layout for the 10th path, tau={tau}")

plt.show()



