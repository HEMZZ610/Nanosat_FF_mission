"""
Typical transformations required for Astrodynamics
    
Date: 11-04-2024

Author:
    SDCS group
    Vishnuvardhan Shakthibala
"""

import numpy
import math

def C_1(theta):
    C=numpy.array([1,0,0],
                  [0,math.cos(theta),math.sin(theta)],
                  [0 -math.sin(theta),math.cos(theta)])
    return C

def C_2(theta):
    C=numpy.array([math.cos(theta),0,-math.sin(theta)],
                  [0,1,0],
                  [math.sin(theta),0,math.cos(theta)])
    return C

def C_3(theta):
    C=numpy.array([math.cos(theta),math.sin(theta),0],
                  [-math.sin(theta),math.cos(theta),0],
                  [0,0,1])
    return C

def PQW2ECI(OM,om,i):
    C=C3(-OM)*C1(-i)*C3(-om)
    return C

def ECI2PQW(OM,om,i):
    C=C3(om)*C1(i)*C3(OM)
    return C

def PQW2RSW(theta):
    C=C3(theta)
    return C

def RSWwPQW(theta):
    C=C3(-theta)
    return C
