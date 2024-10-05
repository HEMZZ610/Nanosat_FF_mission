"""
Nanosat Formation Flying Project

Relative dynamics of two nanosatellites are defined here with J2 perturbation. Taken from paper: 
A planning tool for optimal three-dimensional formation flight maneuvers of satellites in VLEO using aerodynamic lift and drag via yaw angle deviations  
Traub, C., Fasoulas, S., and Herdrich, G. (2022). 

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
import time
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
    NSROE2car,
    NSROE2LVLH,
    NSROE2LVLH_2)

from dynamics import Dynamics_N, yaw_dynamics_N, yaw_dynamics, absolute_NSROE_dynamics, Dynamics



# Parameters that is of interest to the problem

data = {
    "Primary": [3.98600433e5,6378.16,7.2921150e-5],
    "J": [0.1082626925638815e-2, 0, 0],  # J2, J3, J4 coefficients

    # Satellites data including chief and deputies
    "satellites": {
        "chief": {
            "mass": 300,         # Mass in kg
            "area": 2,           # Cross-sectional area in m^2
            "C_D": 0.9,          # Drag coefficient
        },
        "deputy_1": {
            "mass": 250,
            "area": 1.8,
            "C_D": 0.85,
        }
    },
    "N_deputies": 2,  # Number of deputies
    "sat": [1.2, 1.2,1.2],  # Moment of inertia for each satellite

}

print("Parameters initialized.")

deg2rad = numpy.pi / 180

# CHECK Formation Establishment and Reconfiguration Using
# Differential Elements in J2-Perturbed Orbits and SENGUPTA
# Chaser spacecraft initial conditions
# orbital elements - non singular



# Deputy spacecraft relative orbital  elements/ LVLH initial conditions
# NOE_chief = numpy.array([a,lambda_0,i,q1,q2,omega])
NOE_chief = numpy.array([6500,0.1,63.45*deg2rad,0.5,0.2,270.828*deg2rad]) # numpy.array([6803.1366,0,97.04,0.005,0,270.828])
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
alpha = 0#180 * deg2rad  # [rad] - angle between the radial and along-track separation
beta = 0#alpha + 90 * deg2rad # [rad] - angle between the radial and cross-track separation
vd = 0 #-10 # Drift per revolutions m/resolution
d= -1# [m] - along track separation
rho_2 = (2*(eta**2) * d) /(3-eta**2) # [m]  - along-track separation
print("RHO_2",rho_2)
print(d/1+e, d/1-e,  d*(1/(2*(eta**2)) /(3-eta**2)))
parameters=numpy.array([rho_1,rho_2,rho_3,alpha,beta,vd])

print("Formation parameters",parameters)
# Initial relative orbital elements
RNOE_0=Param2NROE(NOE_chief, parameters,data)
# RNOE_0[0]=0
# RNOE_0[2]=-RNOE_0[5]*numpy.cos(NOE_chief[2]) 

# angle of attack for the deputy spacecraft
yaw_1 = 0.12  # [rad] - angle of attack = 0 assumption that V_sat = V_rel
yaw_2 = 0.08  # [rad] - angle of attack = 0
yaw_c_d=numpy.array([yaw_1,yaw_2])

print("RELATIVE ORBITAL ELEMTNS INITIAL", RNOE_0)
print("CHIEF INTIIAL ORBITAL ELEMENTS", NOE_chief)

 
# statement matrix [RNOE_0,NOE_chief,yaw_c_d]
# [6x1,6x1,2x1]
yy_o=numpy.concatenate((RNOE_0,NOE_chief,yaw_c_d))


# test for gauess equation
mu=data["Primary"][0]
Torb = 2*numpy.pi*numpy.sqrt(NOE_chief[0]**3/mu)    # [s]    Orbital period
n_revol_T = 0.0005*365*24*60*60/Torb
n_revolution=  n_revol_T
T_total=n_revolution*Torb

t_span=[0,T_total]
teval=numpy.linspace(0, T_total, 100000)
# K=numpy.array([k1,k2])
 
data["Init"] = [NOE_chief[4],NOE_chief[3], 0]

uu = numpy.zeros((2,1)) # input torque to the dynamics model - it is fed inside the yaw dynamics.

print("Number of Period",n_revolution)
print("Orbital Period",Torb)
print("Time of Integration",T_total)
print("integration time step",teval[1]-teval[0])
print("Number of data points",len(teval))
print("Integration starting....")

# Start the timer
start_time = time.time()

sol=integrate.solve_ivp(Dynamics, t_span, yy_o,t_eval=teval,
                        method='DOP853',args=(data,uu), rtol=1e-10, atol=1e-12,dense_output=True)

# End the timer
end_time = time.time()

# Calculate the time taken
execution_time = end_time - start_time

# Print the execution time
print(f"Time taken for integration: {execution_time:.4f} seconds")

print("Integration done....")

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
    if yy2[1]>2000:
        print("ANOMALY",yy2[1])
    
    rr_s[:,i]=NSROE2LVLH(yy1,yy2,data)

    # h = COE[0]
    # e =COE[1]
    # i =COE[2]
    # OM = COE[3]
    # om =COE[4]
    # TA =COE[5]

    
print("mean position in x",numpy.mean(rr_s[0]))
print("mean position in y",numpy.mean(rr_s[1]))
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
# arrow_length = 0.01  # Adjust this factor to change the relative size of arrows
# a=max(rr_s[0])
# b=max(rr_s[1])
# c= max(rr_s[2])

# d = max([a,b,c])
# # Normalize the vectors based on the axis scales
# x_axis = numpy.array([arrow_length * max(rr_s[0])/d, 0, 0])
# y_axis = numpy.array([0, arrow_length * max(rr_s[1])/d, 0])
# z_axis = numpy.array([0, 0, arrow_length * max(rr_s[2])/d])
# # add xlim and ylim
# ax.set_xlim(-d, d)
# ax.set_ylim(-d, d)

# # x-axis
# ax.quiver(0, 0, 0, x_axis[0], x_axis[1], x_axis[2], color='r', arrow_length_ratio=0.1)
# # y-axis
# ax.quiver(0, 0, 0, y_axis[0], y_axis[1], y_axis[2], color='g', arrow_length_ratio=0.1)
# # z-axis
# ax.quiver(0, 0, 0, z_axis[0], z_axis[1], z_axis[2], color='b', arrow_length_ratio=0.1)
# ax.plot3D(rr_s[0],rr_s[1],rr_s[2] , 'black', linewidth=2, alpha=1)
# ax.set_title('LVLH frame - Deput Spacecraft')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
# The original rr_s array is already defined in your code as the spacecraft trajectory

# Set the limits to 1 km for each axis
x_limits = [-0.5, 1.5]  # 1 km range centered at 0
y_limits = [-0.5, 1.5]  # 1 km range centered at 0
z_limits = [-0.5, 1.5]  # 1 km range centered at 0

# Create a figure with two subplots: one for the interactive 3D plot and one for the dynamic frame
fig = plt.figure(figsize=(12, 6))

# Interactive 3D plot
ax1 = fig.add_subplot(121, projection='3d')

# Plot the trajectory in 3D space
line, = ax1.plot3D(rr_s[0], rr_s[1], rr_s[2], 'black', linewidth=2, alpha=1)

# Draw reference frame arrows (LVLH) on the interactive plot
arrow_length = 0.1  # Adjust this factor to change the relative size of arrows
x_axis = numpy.array([arrow_length, 0, 0])
y_axis = numpy.array([0, arrow_length, 0])
z_axis = numpy.array([0, 0, arrow_length])

# x-axis
x_quiver = ax1.quiver(0, 0, 0, x_axis[0], x_axis[1], x_axis[2], color='r', arrow_length_ratio=0.1)
# y-axis
y_quiver = ax1.quiver(0, 0, 0, y_axis[0], y_axis[1], y_axis[2], color='g', arrow_length_ratio=0.1)
# z-axis
z_quiver = ax1.quiver(0, 0, 0, z_axis[0], z_axis[1], z_axis[2], color='b', arrow_length_ratio=0.1)

# Apply the fixed limits (1 km range) to the interactive plot
ax1.set_xlim(x_limits)
ax1.set_ylim(y_limits)
ax1.set_zlim(z_limits)

# Set axis labels with km units and title
ax1.set_xlabel('x (km)')
ax1.set_ylabel('y (km)')
ax1.set_zlabel('z (km)')
ax1.set_title('LVLH frame - Deput Spacecraft (Interactive)')


# Dynamic frame plot (linked to the interactive plot)
ax2 = fig.add_subplot(122, projection='3d')

# Static reference frame, which will update based on ax1's view
x_quiver2 = ax2.quiver(0, 0, 0, x_axis[0], x_axis[1], x_axis[2], color='r', arrow_length_ratio=0.1)
y_quiver2 = ax2.quiver(0, 0, 0, y_axis[0], y_axis[1], y_axis[2], color='g', arrow_length_ratio=0.1)
z_quiver2 = ax2.quiver(0, 0, 0, z_axis[0], z_axis[1], z_axis[2], color='b', arrow_length_ratio=0.1)

# Set the limits to zoom into 200 meters for each axis (adjust as needed)
x_zoom_limits = [-0.1, 0.1]  # 200 meters range centered at 0
y_zoom_limits = [-0.1, 0.1]  # 200 meters range centered at 0
z_zoom_limits = [-0.1, 0.1]  # 200 meters range centered at 0


# Set axis labels for the dynamic frame plot
ax2.set_xlabel('x (km)')
ax2.set_ylabel('y (km)')
ax2.set_zlabel('z (km)')
ax2.set_title('Dynamic LVLH Frame (Zoomed View)')

# Function to update the dynamic frame based on the interactive plot's view
def update_dynamic_frame(event):
    # Get the current view angle of ax1
    elev = ax1.elev
    azim = ax1.azim

    # Set the same view for ax2 (the dynamic frame)
    ax2.view_init(elev=elev, azim=azim)

    # Redraw the figure to reflect the changes
    fig.canvas.draw_idle()

# Connect the update function to the interactive plot (ax1)
ax1.figure.canvas.mpl_connect('motion_notify_event', update_dynamic_frame)

# Show the zoomed plot
plt.show()

############# Relative orbital Dynamics ####################
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





fig, axs = plt.subplots(3, 1)

# Plot data on the first subplot
axs[0].plot(teval, sol.y[12])
axs[0].set_title('Chief yaw angle')

# Plot data on the second subplot
axs[1].plot(teval, sol.y[13])
axs[1].set_title('Deputy 1 yaw angle')


plt.show()
