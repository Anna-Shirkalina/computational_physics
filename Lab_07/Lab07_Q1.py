"""
Lab7 PHY407
@author: Genevieve Beauregard(primarily), Anna Shirkalina 
Q1a/b/c
b - measures time for adaptive time step
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


def rho(delta, s1, s2, h): 
    """Rho as per the lab handout
    
    Input:
        delta, target accuracy 
        s1, estimate with 2 steps h, s1 = [x1, y1, vx1, vy1]
        s2, estimate with step 2h, s2 = [x2, y2, vx2, vy2]
        h, time step 
    Output:
        Ratio rho, see Eqn 8.53 in newman
    """
    errx = (1/30) * abs(s1[0] - s2[0])
    erry = (1/30) * abs(s1[0] - s2[0])
    # print("the erry term is, "+ str(erry))
    # print("the errx term is," + str(errx))
    # print("the s1 term is " + str(s1))
    # print("the s2 term is" + str(s2))
    
    return (h * delta )/(np.sqrt(errx**2 + erry**2))

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
h= 0.01  # step width
delta = 10e-6

x = [] # empty coordinate lists
y = []
vx = []
vy = []

#RK4 loop

i = 0 # step counter 
time_passed = 0 # time counter 
time_array = []
h_array = []

start = time()
# while loop stops when time => b:
while time_passed < b:
    # append coordinate lists
    x.append(s[0])
    y.append(s[1])
    vx.append(s[2])
    vy.append(s[3])
    
    # Do the two steps of size h
    s1 = s + step(s, h, f)
    s1 = s1 + step(s1, h, f)
    
    # Do 2h step
    s2 = s + step(s, 2*h, f)
    
    
    # obtain ratio
    ratio = rho(delta, s1, s2, h)
    print(h)
    
    # cases 
    if ratio >= 1:
        
        # we keep step 
        s += step(s, h, f)
        
        # update time array
        time_passed += h 
        time_array.append(time_passed)
        
        # update h array with the associated time step
        h_array.append(h)
        
        # change h values 
        h = h*(ratio)**(0.25)
    
    # rho < 1 case
    else: 
        # we need to adjust the value of h till the ratio >= 1 
        h = h*(ratio)**(0.25)
        
        #now we have the update h such that ratio >= 1 
        # do time step 
        s += step(s, h, f)
        time_passed += h 
        time_array.append(time_passed)
        h_array.append(h)
    
end = time()

print("Time the loop take with adaptive h is " + str(end - start) +" seconds") 
    
    
    


# Now we plot y against x, using the same axis

# configuring plots, rica/erik taught me this snippet to format fonts
plt.rc('text', usetex=True)             # use LaTeX for text
plt.rc('font', family='serif')          # use serif font
plt.rcParams.update({'font.size': 15})  # set font size 

plt.figure(figsize=(5,5))
plt.plot(x, y, 'k.')
plt.xlabel('x, distance')
plt.ylabel('y, distance')
plt.title('Garbage Motion with adaptive time step')
plt.savefig('Q1b.pdf')


plt.figure(figsize=(5,5))
plt.plot(time_array, h_array)
plt.xlabel('time(s)')
plt.ylabel('h time step(s)')
plt.title('Time steps against time')
plt.savefig('Q1c.pdf')
