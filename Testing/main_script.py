"""

This script is used to test gauss planetary equation function present inside "TwoBP.py"

Date: 16/08/2024

Author:
    Vishnuvardhan Shakthibala
    
"""
## Copy the following lines of code 
# FROM HERE
import numpy
from scipy import integrate
import matplotlib.pyplot as plt
import os
import sys
import math
## ADD the packages here if you think it is needed and update it in this file.

## Import our libraries here
Library= os.path.join(os.path.dirname(os.path.abspath(__file__)),"../core")
sys.path.insert(0, Library)

from TwoBP import (
    car2kep, 
    kep2car, 
    twobp_cart, 
    gauss_eqn, 
    Event_COE, 
    theta2M, 
    M2theta, 
    Param2NROE, 
    guess_nonsingular_Bmat, 
    lagrage_J2_diff, 
    absolute_NSROE_dynamics ,
    NSROE2car,
    Dynamics,
    NSROE2LVLH)


# Parameters that is of interest to the problem

data={"J":[0.1082626925638815e-2,0,0],"S/C":[300,2,0.9,300],"Primary":[3.98600433e5,6378.16,7.2921150e-5],"sat":[1.2,1.2]}
deg2rad = numpy.pi / 180

# CHECK Formation Establishment and Reconfiguration Using
# Differential Elements in J2-Perturbed Orbits and SENGUPTA
# Chaser spacecraft initial conditions
# orbital elements - non singular



# Deputy spacecraft relative orbital  elements/ LVLH initial conditions
# NOE_chief = numpy.array([a,lambda_0,i,q1,q2,omega])
NOE_chief = numpy.array([6803.1366,0,63.45*deg2rad,0.005,0,270.828*deg2rad]) # numpy.array([6803.1366,0,97.04,0.005,0,270.828])
## MAKE SURE TO FOLLOW RIGHT orbital elements order
 

    # assigning the state variables
a =NOE_chief[0]
l =NOE_chief[1]
i =NOE_chief[2]
q1 =NOE_chief[3]
q2 =NOE_chief[4]
OM =NOE_chief[5]
mu = data["Primary"][0]


e=numpy.sqrt(q1**2 + q2**2)
h=numpy.sqrt(mu*a*(1-e**2))
term1=(h**2)/(mu)
eta = 1- q1**2 - q2**2
p=term1
rp=a*(1-e)
n = numpy.sqrt(mu/(a**3))

if e==0:  
    u = l
    r = (a * eta**2) / (1 + (q1 * numpy.cos(u)) + (q2 * numpy.sin(u)))
else:
    omega_peri = numpy.arccos(q1 / e)
    mean_anamoly = l - omega_peri
    theta_tuple = M2theta(mean_anamoly, e, 1e-8)
    
    theta = theta_tuple[0]
    u = theta + omega_peri
    r = (a * eta**2) / (1 + (q1 * numpy.cos(u)) + (q2 * numpy.sin(u)))

# Design parameters for the formation - Sengupta and Vadali 2007 Relative Motion and the Geometry of Formations in Keplerian Elliptic Orbits

rho_1 = 0 # [m]  - radial separation 
rho_3 =0 # [m]  - cross-track separation
alpha = 180 * deg2rad  # [rad] - angle between the radial and along-track separation
beta = 90 * deg2rad # [rad] - angle between the radial and cross-track separation
vd = 0 #-10 # Drift per revolutions m/resolution

rho_2 = 20 # [m]  - along-track separation

parameters=numpy.array([rho_1,rho_2,rho_3,alpha,beta,vd])

# Initial relative orbital elements
RNOE_0=Param2NROE(NOE_chief, parameters,data)

RNOE_0[2]=-RNOE_0[5]*numpy.cos(NOE_chief[2]) 

# angle of attack for the deputy spacecraft
yaw_1 = 0 
yaw_2 = 0 
yaw_c_d=numpy.array([yaw_1,yaw_2])

print("RELATIVE ORBITAL ELEMTNS INITIAL", RNOE_0)
print("CHIEF INTIIAL ORBITAL ELEMENTS", NOE_chief)

 



# statement matrix [RNOE_0,NOE_chief,yaw_c_d]
# [6x1,6x1,2x1]
yy_o=numpy.concatenate((RNOE_0,NOE_chief,yaw_c_d))


# test for gauess equation
mu=data["Primary"][0]
Torb = 2*numpy.pi*numpy.sqrt(NOE_chief[0]**3/mu)    # [s]    Orbital period
n_revol_T = 24*60*60/Torb
n_revolution= 2*365*n_revol_T
T_total=n_revolution*Torb

t_span=[0,T_total]
teval=numpy.linspace(0, T_total, 20000)
# K=numpy.array([k1,k2])
 
data["Init"] = [NOE_chief[4],NOE_chief[3], 0]

sol=integrate.solve_ivp(Dynamics, t_span, yy_o,t_eval=teval,
                        method='DOP853',args=(data,),rtol=1e-13, atol=1e-10)


# Convert from NROE to Carterian co-ordinates. 



rr_s=numpy.zeros((3,len(sol.y[0])))
vv_s=numpy.zeros((3,len(sol.y[0])))

for i in range(0,len(sol.y[0])):
    # if sol.y[5][i]>2*numpy.pi:
    #     sol.y[5][i]= 
 
    # rr_s[:,i],vv_s[:,i]=NSROE2car(numpy.array([sol.y[0][i],sol.y[1][i],sol.y[2][i],
    #                                            sol.y[3][i],sol.y[4][i],sol.y[5][i]]),data)
    yy1=sol.y[0:6,i]
    yy2=sol.y[6:12,i]
    rr_s[:,i]=NSROE2LVLH(yy1,yy2,data)

    # h = COE[0]
    # e =COE[1]
    # i =COE[2]
    # OM = COE[3]
    # om =COE[4]
    # TA =COE[5]

    


# Spherical earth
# Setting up Spherical Earth to Plot
N = 50
phi = numpy.linspace(0, 2 * numpy.pi, N)
theta = numpy.linspace(0, numpy.pi, N)
theta, phi = numpy.meshgrid(theta, phi)

r_Earth = 6378.14  # Average radius of Earth [km]
X_Earth = r_Earth * numpy.cos(phi) * numpy.sin(theta)
Y_Earth = r_Earth * numpy.sin(phi) * numpy.sin(theta)
Z_Earth = r_Earth * numpy.cos(theta)

# draw the unit vectors of the ECI frame on the 3d plot of earth



# Plotting Earth and Orbit
fig = plt.figure(1)
ax = plt.axes(projection='3d')
# ax.plot_surface(X_Earth, Y_Earth, Z_Earth, color='blue', alpha=0.7)
# x-axis
# Add the unit vectors of the LVLH frame
# Define a constant length for the arrows
# Define a constant arrow length relative to the axis ranges
arrow_length = 0.01  # Adjust this factor to change the relative size of arrows
a=max(rr_s[0])
b=max(rr_s[1])
c= max(rr_s[2])

d = max([a,b,c])
# Normalize the vectors based on the axis scales
x_axis = numpy.array([arrow_length * max(rr_s[0])/d, 0, 0])
y_axis = numpy.array([0, arrow_length * max(rr_s[1])/d, 0])
z_axis = numpy.array([0, 0, arrow_length * max(rr_s[2])/d])
# add xlim and ylim
ax.set_xlim(-d, d)
ax.set_ylim(-d, d)

# x-axis
ax.quiver(0, 0, 0, x_axis[0], x_axis[1], x_axis[2], color='r', arrow_length_ratio=0.1)
# y-axis
ax.quiver(0, 0, 0, y_axis[0], y_axis[1], y_axis[2], color='g', arrow_length_ratio=0.1)
# z-axis
ax.quiver(0, 0, 0, z_axis[0], z_axis[1], z_axis[2], color='b', arrow_length_ratio=0.1)
ax.plot3D(rr_s[0],rr_s[1],rr_s[2] , 'black', linewidth=2, alpha=1)
ax.set_title('LVLH frame - Deput Spacecraft')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')


fig, axs = plt.subplots(3, 1)

# Plot data on the first subplot
axs[0].plot(teval, rr_s[0])
axs[0].set_title('x')

# Plot data on the second subplot
axs[1].plot(teval, rr_s[1])
axs[1].set_title('y')

axs[2].plot(teval, rr_s[2])
axs[2].set_title('z')


fig, axs = plt.subplots(3, 1)

# Plot data on the first subplot
axs[0].plot(teval, sol.y[0])
axs[0].set_title('semi major axis')

# Plot data on the second subplot
axs[1].plot(teval, sol.y[1])
axs[1].set_title('mean true latitude')

axs[2].plot(teval, sol.y[2])
axs[2].set_title('inclination')


fig, axs = plt.subplots(3, 1)

# Plot data on the first subplot
axs[0].plot(teval, sol.y[3])
axs[0].set_title('q1')

# Plot data on the second subplot
axs[1].plot(teval, sol.y[4])
axs[1].set_title('q2')

axs[2].plot(teval, sol.y[5])
axs[2].set_title('right ascenstion of ascending node')



x = rr_s[0]
y = rr_s[1]
z = rr_s[2]
# Plot x and y
plt.figure(5)
plt.plot(x, y, label='x vs y')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Plot of x vs y')


# Plot z and y
plt.figure(6)
plt.plot(z, y, label='z vs y', color='g')
plt.xlabel('z')
plt.ylabel('y')
plt.legend()
plt.title('Plot of z vs y')


# Plot x and z
plt.figure(7)
plt.plot(x, z, label='x vs z', color='r')
plt.xlabel('x')
plt.ylabel('z')
plt.legend()
plt.title('Plot of x vs z')

plt.show()

