"""
Usage:
    This file contains typical perturbations present in space

Author:
    Vishnuvardhan Shakthibala
"""
import numpy
import os
import sys
import math
#from pyatmos import expo


def aspherical_perturbation(rr,data,n):
    # perturbation due to non-spherical earth

    # data={"J":[J2,J3,J4],"S/C":[M_SC,A_cross,C_D,beta],"Primary":[mu,RE.w]}

    # Parameters
    J2=data["J"][0]
    mu=data["Primary"][0]
    R_E=data["Primary"][1]

    r=(rr[0]**2 + rr[1]**2 + rr[2]**2)**0.5 # magnitude

    # Zonal Harmonics upto J6

    # J2 perturbation
    

    a2=numpy.zeros(3)

    term1=(3*J2*mu*R_E**2)/(2*r**4)
    term2=5*(rr[2]**2)/(r**2)
    a2[0]=-term1*(rr[0]/r)*(1-term2)
    a2[1]=-term1*(rr[1]/r)*(1-term2)
    a2[2]=-term1*(rr[2]/r)*(1-term2)



    # we can extend the perturbation terms to include J3, J4, J5 ... effects
    # J3 perturbation

    # a3=numpy.zeros([3,1])

    # term1=(5*J3*mu*(R_E)**3)/(2*r**7)
    # term2=((3*rr(3))-((7*rr(3)**3)/r**2));

    # a3(1,1)=-rr(1,1)*term1*term2;
    # a3(2,1)=-rr(2,1)*term1*term2;
    # a3(3,1)=-term1*((6*rr(3)**2)-((7*rr(3)**4)/r**2)-((3/5)*r**2));


    return a2

def atmosphheric_drag(rr,vv,data):
    # perturbation due to atmospheric drag

    # data={"J":[J2,J3,J4],"S/C":[M_SC,A_cross,C_D,beta],"Primary":[mu,RE.w]}

    # Parameters
    R_E=data["Primary"][1]
    w=data["Primary"][2]

    A=data["S/C"][1]            # [m^2]       Spacecraft cross-sectional area
    m=data["S/C"][0]                # [kg]        Spacecraft mass
    CD=data["S/C"][2]                # [-]     Drag coefficient
    B=data["S/C"][3]

    r=(rr[0]**2 + rr[1]**2 + rr[2]**2)**0.5 # position magnitude
    v_rel=vv-numpy.cross(numpy.array([0,0,w]),rr)
    v_rel_m=(v_rel[0]**2 + v_rel[1]**2 + v_rel[2]**2)**0.5
    # Atmoshphere model
    h=r-R_E
    rho=expo([h])

    # Drag 
    a_d = - 0.5*(1/B)*rho.info['rho'][0]*(v_rel_m**2)*(v_rel/v_rel_m)



    return a_d
