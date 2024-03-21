"""
Title: 
    Gauss planetary equations for evolution of orbital elements
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

    