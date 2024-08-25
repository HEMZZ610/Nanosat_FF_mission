"""
Nanosat Formation Flying Project

Testing the core libraries : Testing the gauss equation
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

from TwoBP import car2kep, kep2car, twobp_cart, gauss_eqn, Event_COE


r=numpy.array([-6045,-3490,2500])
v=numpy.array([-3.457,6.618,2.533])
mu=398600 # gravitational parameter
# test of RVtoOE
COE=car2kep(r,v,mu)

print(COE)

rp = 200                  # [km]        Perigee distance
R0=6378.16

CEO_0=numpy.zeros(6)

# orbital elements [a,e,i,RAAN,omega, theta]
CEO_0[1] = 0.0045                 # [-]         Eccentricity
CEO_0[0]=( rp + R0 ) / ( 1 + CEO_0[1])  
CEO_0[2] = numpy.pi / 2                 # [rad]       Orbit inclination
CEO_0[3] = 0.0001             # [rad]       Orbit RAAN
CEO_0[4] = 0                  # [rad]       Orbit argument of perigee
CEO_0[5] = 0                  # [rad]       Orbit initial true anomaly
h0 = numpy.sqrt(mu*CEO_0[0]*(1-CEO_0[1]**2))  # [km]    Angular momentum axis of the orbit



Torb = 2*numpy.pi*numpy.sqrt(CEO_0[0]**3/mu)    # [s]    Orbital period
n_revolution=100
T_total=n_revolution*Torb

# test for gauess equation

r0=numpy.concatenate((r,v)) 
t_span=[0,T_total]
teval=numpy.linspace(0, T_total, 20000)
# K=numpy.array([k1,k2])

sol=integrate.solve_ivp(gauss_eqn, t_span, CEO_0,t_eval=teval,
                        method='RK23',args=(mu,),rtol=1e-8, atol=1e-6,events=[Event_COE])

rr_s=numpy.zeros((3,len(sol.y[0])))
vv_s=numpy.zeros((3,len(sol.y[0])))

for i in range(0,len(sol.y[0])):
    if sol.y[5][i]>2*numpy.pi:
        sol.y[5][i]=sol.y[5][i]-2*numpy.pi     
    h = numpy.sqrt(mu*sol.y[0][i]*(1-sol.y[1][i]**2)) 
    rr_s[:,i],vv_s[:,i]=kep2car(numpy.array([h,sol.y[1][i],sol.y[2][i],sol.y[3][i],sol.y[4][i],sol.y[5][i]]),mu)


    


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

# Plotting Earth and Orbit
fig = plt.figure(1)
ax = plt.axes(projection='3d')
ax.plot_surface(X_Earth, Y_Earth, Z_Earth, color='blue', alpha=0.7)
# plotting
ax.plot3D(rr_s[0],rr_s[1],rr_s[2] , 'black', linewidth=2, alpha=1)
ax.set_title('two body trajectory')




# fig, axs = plt.subplots(3, 1)

# # Plot data on the first subplot
# axs[0].plot(teval, rr_s[0])
# axs[0].set_title('x')

# # Plot data on the second subplot
# axs[1].plot(teval, rr_s[1])
# axs[1].set_title('y')

# axs[2].plot(teval, rr_s[2])
# axs[2].set_title('z')


fig, axs = plt.subplots(3, 1)

# Plot data on the first subplot
axs[0].plot(teval, sol.y[0])
axs[0].set_title('semi major axis')

# Plot data on the second subplot
axs[1].plot(teval, sol.y[1])
axs[1].set_title('eccentricity')

axs[2].plot(teval, sol.y[2])
axs[2].set_title('inclination')


fig, axs = plt.subplots(3, 1)

# Plot data on the first subplot
axs[0].plot(teval, sol.y[3])
axs[0].set_title('RAAN')

# Plot data on the second subplot
axs[1].plot(teval, sol.y[4])
axs[1].set_title('omega')

axs[2].plot(teval, sol.y[5])
axs[2].set_title('True Anamoly')

plt.show()

