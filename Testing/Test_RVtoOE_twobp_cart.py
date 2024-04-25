"""
Usage:
    

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

from TwoBP import car2kep, twobp_cart

r=numpy.array([-6045,-3490,2500])
v=numpy.array([-3.457,6.618,2.533])
mu=398600 # gravitational parameter
# test of RVtoOE
COE=car2kep(r,v,mu)

print(COE)
CEO_0=numpy.zeros((6,1))
rp = 254.9 ;                 # [km]        Perigee distance
CEO_0[] = 0.0045 ;                # [-]         Eccentricity
data.orbit.a0 = ( data.orbit.rp + data.orbit.Re0 ) / ( 1 - data.orbit.e0 ) ;  # [km]    Semimajor axis of the orbit
data.orbit.i0 = pi / 2 ;                # [rad]       Orbit inclination
data.orbit.Omega0 = 0.0001 ;            # [rad]       Orbit RAAN
data.orbit.omega0 = 0 ;                 # [rad]       Orbit argument of perigee
data.orbit.theta0 = 0 ;                 # [rad]       Orbit initial true anomaly
data.orbit.Torb = 2*pi*sqrt(data.orbit.a0^3/data.orbit.mu) ;   # [s]    Orbital period


# test for 2body caresian function twobp_cart

r0=numpy.concatenate((r,v)) 
t_span=[0,8203.68]
teval=numpy.linspace(0, 8203.68, 200)
# K=numpy.array([k1,k2])

sol=integrate.solve_ivp(twobp_cart, t_span, r0, method='RK23',t_eval=teval,args=(mu,))


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
ax.plot3D(sol.y[0],sol.y[1],sol.y[2] , 'black', linewidth=2, alpha=1)
ax.set_title('two body trajectory')
plt.show()