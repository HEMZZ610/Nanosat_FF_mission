"""
Standard transformations required for Astrodynamics
    
Date: 25-04-2024

Author:
    SDCS group
    Vishnuvardhan Shakthibala
"""

import numpy
import math

def C1(theta):
    C=numpy.array([[1,0,0],
                  [0,numpy.cos(theta),numpy.sin(theta)],
                  [0,-numpy.sin(theta),numpy.cos(theta)]])
    return C

def C2(theta):
    C=numpy.array([[numpy.cos(theta),0,-numpy.sin(theta)],
                  [0,1,0],
                  [numpy.sin(theta),0,numpy.cos(theta)]])
    return C

def C3(theta):
    C=numpy.array([[numpy.cos(theta),numpy.sin(theta),0],
                  [-numpy.sin(theta),numpy.cos(theta),0],
                  [0,0,1]])
    return C

# references frame transformations

# Orbit Perifocal to Earth centric inertial frame
def PQW2ECI(OM,om,i):
    C=numpy.matmul(C3(-OM),numpy.matmul(C1(-i),C3(-om)))
    return C

# Earth centric inertial to Orbit Perifocal
def ECI2PQW(OM,om,i):
    C=C3(om)*C1(i)*C3(OM)
    return C

# Orbit Perifocal to RSW (LVLH) frame
def PQW2RSW(theta):
    C=C3(theta)
    return C

# RSW (LVLH) frame to Orbit Perifocal
def RSW2PQW(theta):
    C=C3(-theta)
    return C

def RSW2ECI(OM,om,i,theta):
    C=numpy.matmul(C3(-OM),numpy.matmul(C1(-i),C3(-(om+theta))))
    u=om+theta
    C_org=numpy.array([[-numpy.sin(OM)*numpy.sin(i)*numpy.sin(u)+numpy.sin(OM)*numpy.sin(u),numpy.sin(OM)*numpy.sin(i)*numpy.sin(u)+numpy.sin(OM)*numpy.sin(u),numpy.sin(i)*numpy.sin(u)],
        [-numpy.sin(OM)*numpy.sin(i)*numpy.sin(u)-numpy.sin(OM)*numpy.sin(u),numpy.sin(OM)*numpy.sin(i)*numpy.sin(u)-numpy.sin(OM)*numpy.sin(u),numpy.sin(i)*numpy.sin(u)],
        [numpy.sin(OM)*numpy.sin(i),-numpy.sin(OM)*numpy.sin(i),numpy.sin(i)]])


    return C_org


