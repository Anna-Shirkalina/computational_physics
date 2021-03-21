"""
Lab7b PHY407
@author: Genevieve Beauregard(primarily), Anna Shirkalina 
Q1b, segment from lab06 with timing 
"""
import numpy as np 
import numpy.fft as fft 
import matplotlib.pyplot as plt
from time import time


G = 1
M = 10
L = 2

def f(s):
    """ODE to be solved numerically via RK4. Returns functions to be solved 
    simulateanously. 
    Input: 
        r = [x, y, v_x, v_y]
        t, time function evaluated
        G, M, L constants
    Output: [vx,vy,fx,fy], simulataneous first order ODEs
    """
    # isolating variables    
    x = s[0]
    y = s[1]
    vx = s[2]
    vy = s[3]

    # obtaining r 
    r = np.sqrt(x**2 + y**2)
    
    # defining components
    fx = -G*M *(x)/(r**2 * np.sqrt(r**2 + (L**2)/4))
    fy = -G*M * (y)/(r**2 * np.sqrt(r**2 + (L**2)/4))
 
    
    return np.array([vx, vy, fx, fy])


def step(s, h, f):
    """Calculates the step at a data point array s, for a step size h 
    and a function f 
    """
    # get the k1, k2, k3, k4
    k1 = np.multiply(f(s),h)
    k2 = np.multiply(f(s + 0.5*k1),h)
    k3 = np.multiply(f(s + 0.5*k2),h)
    k4 = np.multiply(f(s + k3),h)
    
    return (k1 + 2*k2 + 2*k3 + k4)/6
    
    


# Define constants and initial conditions

# initital [x,y, vx, vy]
s = np.array([1,0,0,1], dtype=float)


a= 0.0      # start of the interval 
b= 10.0     # end of the interval 
N= 10000     # number of steps
h= (b-a)/N  # step width

s_array = np.zeros((N,4)) # empty array for s values
x = [] # empty coordinate lists
y = []
vx = []
vy = []

start = time()
#RK4 loop
for i in range(N): 
    # append coordinate lists
    x.append(s[0])
    y.append(s[1])
    vx.append(s[2])
    vy.append(s[3])
    
    # # get the k1, k2, k3, k4
    # k1 = np.multiply(f(s),h)
    # k2 = np.multiply(f(s + 0.5*k1),h)
    # k3 = np.multiply(f(s + 0.5*k2),h)
    # k4 = np.multiply(f(s + k3),h)
    
    # update s value
    s += step(s, h, f)

end = time ()

print("Rime it takes is, for fixed h, " + str(end - start) + "seconds")
# Now we plot y against x, using the same axis

# configuring plots, rica/erik taught me this snippet to format fonts
plt.rc('text', usetex=True)             # use LaTeX for text
plt.rc('font', family='serif')          # use serif font
plt.rcParams.update({'font.size': 15})  # set font size 

plt.figure(figsize=(5,5))
plt.plot(x, y, '.')
plt.xlabel('x, distance')
plt.ylabel('y, distance')
plt.title('Garbage Motion')
plt.savefig('Q1b.pdf')
