"""
Nanosat Formation Flying Project

All the functions\ required for frame transformations are defined here

Author:
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


# Add frames here

# LVLH FRAME
def LVLHframe(rr,vv):
    # rr and vv are in ECI frame
    x_unit=rr / (rr[0]**2 + rr[1]**2 + rr[2]**2)**0.5
    z_unit=numpy.cross(rr,vv)/numpy.linalg.norm(numpy.cross(rr,vv))
    y_unit=numpy.cross(z_unit,x_unit)

    #print("\n",numpy.linalg.norm(r_unit),"|",numpy.linalg.norm(w_unit),"|",numpy.linalg.norm(s_unit))
    # t_h = vv / numpy.linalg.norm( vv )                                          # velocity versoe
    # h_h = numpy.cross( rr, vv ) / numpy.linalg.norm(numpy.cross(rr, vv ) )           # angular momentum veror
    # n_h = numpy.cross( h_h, t_h ) 


    # matrix to convert from IJK frame to RSW frame
    Rot_LVLH = numpy.vstack([x_unit, y_unit, z_unit])
    return Rot_LVLH

# FRENET FRAME
def Frenetframe(rr,vv):
    # rr and vv are in ECI frame
    T_unit=vv / (vv[0]**2 + vv[1]**2 + vv[2]**2)**0.5
    W_unit=numpy.cross(rr,vv)/numpy.linalg.norm(numpy.cross(rr,vv))
    N_unit=numpy.cross(T_unit,W_unit)

    #print("\n",numpy.linalg.norm(r_unit),"|",numpy.linalg.norm(w_unit),"|",numpy.linalg.norm(s_unit))
    # t_h = vv / numpy.linalg.norm( vv )                                          # velocity versoe
    # h_h = numpy.cross( rr, vv ) / numpy.linalg.norm(numpy.cross(rr, vv ) )           # angular momentum veror
    # n_h = numpy.cross( h_h, t_h ) 


    # matrix to convert from IJK frame to RSW frame
    Rot_FrenetFrame = numpy.vstack([T_unit, N_unit, W_unit])
    return Rot_FrenetFrame

# BODY FIXED FRAME


