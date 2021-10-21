"""
Dumping Truck Analysis: Newmark Integration
Kinematics and Dynamics of Mechanisms: Practical Work 1
"""
### Code Diverges: Mostly because of some problem in the equations implemented > ?
### Initial conditions = Perfect
### Polynomial = Plot function for the polynomial coefficients show good results of 3 order

# ----------------------------------------------------------------------------------------------------------------------------------------
import math
import sys

import numpy as np
from math import sin, cos
from numpy.linalg import norm
from matplotlib import pyplot as plt

#------------------------------------------------------------------------------------------------------------------------------------------

# Parameters and Initial Conditions

# Length of actuator
Lmin = 0.924
Lmax = 1.15 * Lmin

# Masses of Rigid Links
m1 = m4 = 3
m2 = m3 = 15
m5 = 300

# Lengths of Rigid Links
l1 = l4 = 0.4
l2 = l3 = 2
l5 = 1.6

# Gravity
g = 9.81

# Initial Conditions (q0, dq0, ddq0): Values taken from Solidworks model when bar 3 and 5 are horizontal
phi8 = phi14 = 0  # Angles in set of active coordinates
phi1 = 10.7583 * math.pi / 180
phi11 = 7.9703 * math.pi / 180
phi4 = 5.9363 * math.pi / 180

theta = 13.0798 * math.pi / 180  # Angle of driving constraint (Actuator): Additional coordinate introduced to represent actuated joint

n = 26    # Number of active coordinates
m = 26    # Number of constraints

# Coordinate Vector: q = [phi1, x2, y2, x3, y3, phi4, x5, y5, x6, y6, x7, y7, phi8, x9, y9, x10, y10, phi11, x12, y12, x13, y13, phi14, x15, y15, theta].T
q0 = np.array([phi1, 
               0.5*l1*cos(phi1),
               0.5*l1*sin(phi1), 
               l1*cos(phi1), 
               l1*sin(phi1), 
               phi4, 
               l1*cos(phi1)+0.5*l2*cos(phi4), 
               l1*sin(phi1)+0.5*l2*sin(phi4), 
               l1*cos(phi1)+0.65*l2*cos(phi4), 
               l1*sin(phi1)+0.65*l2*sin(phi4), 
               l1*cos(phi1)+l2*cos(phi4), 
               l1*sin(phi1)+l2*sin(phi4), 
               phi8, 
               1.6-0.5*l3*cos(phi8), 
               0.2+0.5*l3*sin(phi8), 
               1.6-l3*cos(phi8), 
               0.2+l3*sin(phi8), 
               phi11, 
               1.6-l3*cos(phi8)+0.5*l4*cos(phi11), 
               0.2+l3*sin(phi8)+0.5*l4*sin(phi11), 
               1.6-l3*cos(phi8)+l4*cos(phi11), 
               0.2+l3*sin(phi8)+l4*sin(phi11), 
               phi14, 
               1.6-l3*cos(phi8)+l4*cos(phi11)+0.5*l5*cos(phi14), 
               0.2+l3*sin(phi8)+l4*sin(phi11)-0.5*l5*sin(phi14), 
               theta])

# Newmark Parameters (Numerical Damping)
alpha = 0.015
gamma = 0.5 + alpha
beta = 0.25 * (0.5 + gamma) ** 2

# Integration Parameters
h = 0.01  # Time step size
tf = 15   # Total simulation time
n_steps = round(tf / h)
tol = 1e-6
niter_max = 20
nm_tot = n + m

#--------------------------------------------------------------------------------------------------------------------------
# Function Definitions

def calculate_ft(t):
    D = Lmin
    C = 0
    B = (3 * (Lmax - Lmin) / tf ** 2)
    A = (-2 / 3 * (B / tf))

    polynom = A * t ** 3 + B * t ** 2 + C * t + D

    return polynom

def mass_matrix():
    M = np.diag([0,m1,m1,0,0,0,m2,m2,0,0,0,0,0,m3,m3,0,0,0,m4,m4,0,0,0,m5,m5,0])
    return M


def ext_force():
    f_ext = np.array([[0,0,-m1*g,0,0,0,0,-m2*g,0,0,0,0,0,0,-m3*g,0,0,0,0,-m4*g,0,0,0,0,-m5*g,0]]).transpose()
    return f_ext


def stiffness_matrix(q, lmbda):
    Kt = np.zeros((n,n))

    Kt[0, 0] = - (0.5 * l1 * cos(q[0]) * lmbda[0]) + (0.5 * l1 * sin(q[0]) * lmbda[1]) - (l1 * cos(q[0]) * lmbda[2]) + (l1 * sin(q[0]) * lmbda[3])
    Kt[5, 5] = (0.5 * l2 * cos(q[5]) * lmbda[4]) + (0.5 * l2 * sin(q[5]) * lmbda[5]) + (0.65 * l2 * cos(q[5]) * lmbda[6]) + (0.65 * l2 * sin(q[5]) * lmbda[7]) + (l2 * cos(q[5]) * lmbda[8]) + (l2 * sin(q[5]) * lmbda[9])
    Kt[12, 12] = (-0.35 * l3 * cos(q[12]) * lmbda[10]) + (0.35 * l3 * sin(q[12]) * lmbda[11]) - (0.5 * l3 * cos(q[12]) * lmbda[12]) + (0.5 * l3 * sin(q[12]) * lmbda[13]) - (l3 * cos(q[12]) * lmbda[14]) + (l3 * sin(q[12]) * lmbda[15])
    Kt[17, 17] = (0.5 * l4 * cos(q[17]) * lmbda[16]) + (0.5 * l4 * sin(q[17]) * lmbda[17]) + (l4 * cos(q[17]) * lmbda[18]) + (l4 * sin(q[17]) * lmbda[19])
    Kt[22, 22] = (0.5 * l5 * cos(q[22]) * lmbda[20]) - (0.5 * l5 * sin(q[22]) * lmbda[21]) + (l5 * cos(q[22]) * lmbda[22]) + (l5 * sin(q[22]) * lmbda[23])
    Kt[25, 25] = (ft * cos(theta) * lmbda[24]) + (ft * sin(theta) * lmbda[25])

    return Kt


def constraint_gradient(q_vec):
    G = np.zeros((m, n))

    G[0,0] = - 0.5 * l1 * sin(q_vec[0])
    G[0,1] = 1
    G[1,0] = - 0.5 * l1 * cos(q_vec[0])
    G[1,2] = 1
    G[2,0] = - l1 * sin(q_vec[0])
    G[2,3] = 1
    G[3,0] = - l1 * cos(q_vec[0])
    G[3,4] = 1
    G[4,3] = -1
    G[4,5] = 0.5 * l2 * sin(q_vec[5])
    G[4,6] = 1
    G[5,4] = -1
    G[5,5] = -0.5 * l2 * cos(q_vec[5])
    G[5,7] = 1
    G[6,3] = -1
    G[6,5] = 0.65 * l2 * sin(q_vec[5])
    G[6,8] = 1
    G[7,4] = -1
    G[7,5] = -0.65 * l2 * cos(q_vec[5])
    G[7,9] = 1
    G[8,3] = -1 
    G[8,5] = l2 * sin(q_vec[5])
    G[8,10] = 1
    G[9,4] = -1
    G[9,5] = -l2 * cos(q_vec[5])
    G[9,11] = 1
    G[10,8] = 1 
    G[10,12] = -0.35 * l3 * sin(q_vec[12])
    G[11,9] = 1
    G[11,12] = -0.35 * l3 * cos(q_vec[12])
    G[12,12] = -0.5 * l3 * sin(q_vec[12])
    G[12,13] = 1
    G[13,12] = -0.5 * l3 * cos(q_vec[12])
    G[13,14] = 1
    G[14,12] = -l3 * sin(q_vec[12])
    G[14,15] = 1
    G[15,12] = -l3 * cos(q_vec[12])
    G[15,16] = 1
    G[16,15] = -1
    G[16,17] = 0.5 * l4 * sin(q_vec[17])
    G[16,18] = 1
    G[17,16] = -1
    G[17,17] = -0.5 * l4 * cos(q_vec[17])
    G[17,18] = 1
    G[18,15] = -1 
    G[18,17] = l4 * sin(q_vec[17])
    G[18,20] = 1
    G[19,16] = -1
    G[19,17] = -l4 * cos(q_vec[17])
    G[19,21] = 1
    G[20,20] = -1
    G[20,22] = 0.5 * l5 * sin(q_vec[22])
    G[20,23] = 1
    G[21,21] = -1
    G[21,22] = 0.5 * l5 * cos(q_vec[22])
    G[21,24] = 1
    G[22,10] = 1
    G[22,20] = -1
    G[22,22] = l5 * sin(q_vec[22])
    G[23,11] = 1
    G[23,21] = -1
    G[23,22] = -l5 * cos(q_vec[22])
    G[24,8] = 1
    G[24,25] = ft * sin(q_vec[25])
    G[25,9] = 1
    G[25,25] = -ft * cos(q_vec[25])

    return G


def residual(q_vec, ddq_vec, lmbda_vec):
    M = mass_matrix()
    f_ext = ext_force()
    G = constraint_gradient(q_vec)
    G_T = G.transpose()
    g_q = np.zeros((m, 1))
    g_q = np.array([q_vec[1] + 0.5 * l1 * cos(q_vec[0]),
                    q_vec[2] - 0.5 * l1 * sin(q_vec[0]),
                    q_vec[3] + l1 * cos(q_vec[0]),
                    q_vec[4] - l1 * sin(q_vec[0]),
                    q_vec[6] - q_vec[3] - 0.5 * l2 * cos(q_vec[5]),
                    q_vec[7] - q_vec[4] - 0.5 * l2 * sin(q_vec[5]),
                    q_vec[8] - q_vec[3] - 0.65 * l2 * cos(q_vec[5]),
                    q_vec[9] - q_vec[4] - 0.65 * l2 * sin(q_vec[5]),
                    q_vec[10] - q_vec[3] - l2 * cos(q_vec[5]),
                    q_vec[11] - q_vec[4] - l2 * sin(q_vec[5]),
                    q_vec[8] - 1.6 + 0.35 * l3 * cos(q_vec[12]),
                    q_vec[9] - 0.2 - 0.35 * l3 * sin(q_vec[12]),
                    q_vec[13] - 1.6 + 0.5 * l3 * cos(q_vec[12]),
                    q_vec[14] - 0.2 - 0.5 * l3 * sin(q_vec[12]),
                    q_vec[15] - 1.6 + l3 * cos(q_vec[12]),
                    q_vec[16] - 0.2 - l3 * sin(q_vec[12]),
                    q_vec[18] - q_vec[15] - 0.5 * l4 * cos(q_vec[17]),
                    q_vec[19] - q_vec[16] - 0.5 * l4 * sin(q_vec[17]),
                    q_vec[20] - q_vec[15] - l4 * cos(q_vec[17]),
                    q_vec[21] - q_vec[16] - l4 * sin(q_vec[17]),
                    q_vec[23] - q_vec[20] - 0.5 * l5 * cos(q_vec[22]),
                    q_vec[24] - q_vec[21] + 0.5 * l5 * sin(q_vec[22]),
                    q_vec[10] - q_vec[20] - l5 * cos(q_vec[22]),
                    q_vec[11] - q_vec[21] - l5 * sin(q_vec[22]),
                    q_vec[8] - ft * cos(q_vec[25]),
                    q_vec[9] - ft * sin(q_vec[25])])
    
    R = np.concatenate((M @ ddq_vec + G_T @ lmbda_vec - f_ext, g_q))

    return R


def iteration_matrix(q_vec, lmbda_vec):
    St = np.zeros((nm_tot, nm_tot))
    G = constraint_gradient(q_vec)
    G_T = G.transpose()
    Kt = stiffness_matrix(q_vec, lmbda_vec)
    M = mass_matrix()

    St[:n, :n] = (1 / (beta * h ** 2)) * M + Kt
    St[:n, n:nm_tot] = G_T
    St[n:nm_tot, :n] = G

    return St


# ----------------------------------------------------------------------------------------------------------------------------------------


# Newmark Integration
q = np.empty((n, n_steps+1))
dq = np.empty((n, n_steps+1))
ddq = np.empty((n, n_steps+1))
lmbda = np.empty((m, n_steps+1))

# Initialize variables with initial state 
q[:, 0] = q0
dq[:, 0].fill(0)
ddq[:, 0].fill(0)
lmbda[:, 0].fill(0)

step = 0

while step < n_steps:
    step += 1

    print(f"Step {step} of {n_steps}")

    i = range(step, step+1)
    i_1 = range(step-1, step)

    ft = calculate_ft(h * step)

    # Predictor (Initial Guess) 
    dq[:, i] = dq[:, i_1] + (1 - gamma) * h * ddq[:, i_1]
    q[:, i] = q[:,i_1] + h * dq[:, i_1] + (0.5 - beta) * h ** 2 * ddq[:, i_1]

    # Calculate residual for predicted value (initial guess)
    niter = 0
    
    while niter <= niter_max:
        res = residual(q[:, i], ddq[:, i], lmbda[:, i])
        
        if norm(res) <= tol:
            break

        # if niter == niter_max:
        #     print("Code diverging")
        #     sys.exit()

    # Calculate corrections with iteration matrix (Jacobian): If not under Tolerance.
        St = iteration_matrix(q[:, i], lmbda[:, i])
        delta_total = -np.linalg.solve(St, res)
        delta_q = delta_total[0:n, :]
        delta_lambda = delta_total[n:nm_tot, :]

    # Incrementation of corrections
        q[:, i] += delta_q
        dq[:, i] += gamma / beta / h * delta_q
        ddq[:, i] += 1 / (beta * h ** 2) * delta_q
        lmbda[:, i] += delta_lambda 

        niter += 1

time_array = np.append(np.arange(0, tf, h), [tf])
plt.plot(time_array, q[24, :])
plt.show()