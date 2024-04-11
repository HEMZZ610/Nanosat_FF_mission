"""
Usage:
    

Author:
    Vishnuvardhan Shakthibala
"""
import numpy
from scipy import integrate,optimize
import matplotlib.pyplot as plt
import os
import sys
import math

# conversion factor
Rad2deg=180/math.pi


def RVtoOE(r,v,mu):

    # Cartesian to orbital elements convertion function
    
    # magnitude of position
    r_m=numpy.sqrt(r[0]**2+r[1]**2+r[2]**2)
    
    # magnitude of velocity
    v_m=numpy.sqrt(v[0]**2+v[1]**2+v[2]**2)
    
    # specific angular momentum
    h=numpy.cross(r,v)
    
    # magnitude of specific angular momentum
    h_m=numpy.sqrt(h[0]**2+h[1]**2+h[2]**2)

    # Inclincation i
    i=numpy.arccos(h[2]/h_m)
    
    # Lines of nodes vector N
    K=[0,0,1]
    N=numpy.cross(K,h)
    
    N_m=numpy.sqrt(N[0]**2+N[1]**2+N[2]**2)

    #right ascension of ascending node
    Omega_temp=numpy.arccos(N[0]/N_m)
    
    if N[1]>=0:
        Omega=Omega_temp
    elif N[1]<0:
        Omega=2*numpy.pi-Omega_temp     
    
    
    # eccentricity vector
    
    e=(1/mu)*(numpy.cross(v,h)-mu*(r/r_m))
    
    e_m=numpy.sqrt(e[0]**2+e[1]**2+e[2]**2)
    
    # argument of periapsis
    
    omega_temp=numpy.arccos(numpy.dot((N/N_m),(e/e_m)))
    
    if e[2]>=0:
        omega=omega_temp
    elif e[2]<0:
        omega=2*numpy.pi-omega_temp 
        
    # true anomaly
    
    theta_temp=numpy.arccos(numpy.dot(e,r)/(e_m*r_m))
    
    if numpy.dot(r,v)>=0:
        theta=theta_temp
    elif numpy.dot(r,v)<0:
        theta=2*numpy.pi-theta_temp 
    
    COE_vec=numpy.array([h[0],h[1],h[2],e[0],e[1],e[2],i,Omega,omega,theta])
    COE_mag=numpy.array([h_m,e_m,Rad2deg*i,Rad2deg*Omega,Rad2deg*omega,Rad2deg*theta])
    # angular momentum, eccentricity, inclination, RAAN, argument of perigee, true anomaly
    COE=(COE_vec,COE_mag)
    
    return COE

def twobp_cart(t,r0,mu):
    # Ideal two body problem in cartesian

    dy_dt=numpy.zeros((6,))
    r_m=numpy.sqrt(r0[0]**2+r0[1]**2+r0[2]**2)
    dy_dt[0]=r0[3]
    dy_dt[1]=r0[4]
    dy_dt[2]=r0[5]

    dy_dt[3]=-(mu/(r_m**3))*r0[0]
    dy_dt[4]=-(mu/(r_m**3))*r0[1]
    dy_dt[5]=-(mu/(r_m**3))*r0[2]
    
    return dy_dt

def fun_timeTotheta(x,COE,Me):
    # 
    x=Me-COE[1]*numpy.sin(x)
    return x


def OE_Thetatotime(COE,T,mu):

    term1=numpy.sqrt((1-COE[1]/1+COE[1])*numpy.tan(COE[5]/2))
    E = 2 * numpy.arctan(term1)
    
    Me=E-COE[1]*numpy.sin(E)
    
    t=(Me*T/(2*numpy.pi))

    return t

def OE_Timetotheta(COE,T,mu):

    E = optimize.fsolve(fun_timeTotheta,0.2,COE)
    term1=numpy.sqrt((1+COE[1]/1-COE[1])*numpy.tan(E/2))
    theta = 2 * numpy.arctan(term1)
    return theta

def J2_Perturb(t,x0,param):
    # Guass planetary equations
    # Input, 
    # t - time
    # x0 - state vector
    # param is a tuple 3 x 1
    # param[0] - COE vector
    # param[1] - J2 constant value
    # param[0] - list of information related to Earth [mu, radius]

    dy_dt=numpy.zeros((6,))
    COE=param[0]
    J2=param[1]
    mu=param[2][0]
    R=param[2][1]
    # state vector [h,e,theta,RAAN,i,omega]
    r=((x0[0]**2)/mu)*(1/(1+x0[1]*numpy.cos(x0[2])))
    u=x0[5]+x0[2]
    
    term2=(numpy.sin(x0[4])**2)*numpy.sin(2*u)
    term1=(3/2)*((J2*mu*R**2)/(r**3))
    dy_dt[0]=term1*term2
    
    # eccentricy
    dy_dt[1]=term1*(((x0[0]**2)/(mu*r))*
            numpy.sin(x0[2])*((3*numpy.sin(x0[4])**2)*(numpy.sin(u)**2)-1)
             -numpy.sin(2*u)*(numpy.sin(x0[4])**2)
             *((3+x0[1]*numpy.cos(x0[5]))*numpy.cos(x0[5])+x0[1]))
    
    # True anomaly
    
    dy_dt[2]=(x0[0]/r**2)+((1/(x0[0]*x0[1]))*term1)*(((x0[0]**2)/(mu*r))*
            numpy.cos(x0[2])*((3*numpy.sin(x0[4])**2)*(numpy.sin(u)**2)-1)
             -((2+x0[1]*numpy.cos(x0[5]))*numpy.sin(2*u)*(numpy.sin(x0[4])**2)*numpy.sin(x0[5])))

    # RAAN
    term1=-3*((J2*mu*R**2)/(x0[0]*r**3))
    dy_dt[3]=term1*(numpy.sin(u)**2)*numpy.sin(x0[4])
    
    # inclination

    term1=-(3/4)*((J2*mu*R**2)/(x0[0]*r**3))
    
    dy_dt[4]=term1*numpy.sin(2*u)*numpy.sin(2*x0[4])

    # argument of periapsis
    term1=(3/2)*((J2*mu*R**2)/(x0[1]*x0[0]*r**3))
    dy_dt[5]=term1*(((x0[0]**2)/(mu*r))*
            numpy.cos(x0[5])*((1-3*numpy.sin(x0[4])**2)*(numpy.sin(u)**2))
             -(2+x0[1]*numpy.cos(x0[5]))*numpy.sin(2*u)*(numpy.sin(x0[4])**2)
             *numpy.cos(x0[5])+2*x0[1]*(numpy.cos(x0[5])**2)*(numpy.sin(u)**2))
    
    
    return dy_dt


# References frames and convertions