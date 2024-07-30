"""
This file contains functions required for orbital mechanics - starting from 2body problem to gauss planetary equations + auxillary funtions

Date: 25-04-2024

Author:
    Vishnuvardhan Shakthibala
"""
import numpy
from scipy import integrate,optimize
import matplotlib.pyplot as plt
import os
import sys
import math

from Transformations import PQW2ECI,RSW2ECI
from Perturbations import aspherical_perturbation,atmosphheric_drag

# conversion factor
Rad2deg=180/math.pi


def car2kep(r,v,mu):

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
    # orbital elements = [angular momentum, eccentricity, inclination, RAAN, argument of perigee, true anomaly]
    COE=(COE_vec,COE_mag)
    
    return COE

def kep2car(COE,mu):
    h = COE[0]
    e =COE[1]
    i =COE[2]
    OM = COE[3]
    om =COE[4]
    TA =COE[5]
    
    # obtaining position and velocity vector in perifocan reference frame
    rp = ( h ** 2 / mu ) * ( 1 / ( 1 + e * numpy.cos( TA ) ) ) * ( numpy.cos( TA ) * numpy.array([ 1 , 0 ,0 ])
        + numpy.sin( TA ) * numpy.array([ 0, 1, 0 ]) )
    vp = ( mu / h ) * ( -numpy.sin( TA ) * numpy.array([ 1 , 0 , 0 ]) + ( e + numpy.cos( TA ) ) * numpy.array([ 0 , 1 , 0 ]) ) ;


    RR=numpy.matmul(PQW2ECI(OM,om,i),numpy.transpose(rp))
    VV=numpy.matmul(PQW2ECI(OM,om,i),numpy.transpose(vp))

    return (RR,VV)



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

def M2theta(M,e,tol):

    if M < numpy.pi :
        E0=M+e/2
    elif M >numpy.pi:
        E0=M- e/2
    toll_diff =10
    while (toll_diff>tol):
        term1 = E0-e*numpy.sin(E0)-M
        term2 = 1-e*numpy.cos(E0)
        E2=E0 - (term1/term2)
        toll_diff = abs(E2-E0)
        E0=E2

    theta = 2*numpy.arctan(numpy.tan(E2/2)*numpy.sqrt((1+e)/(1-e)))
    return  (theta,theta* (180/numpy.pi))

def theta2M(theta,e):

    if theta > 2*numpy.pi :
        theta = theta- 2*numpy.pi

    E = 2*numpy.arctan(numpy.tan(theta/2)*numpy.sqrt((1-e)/(1+e)))
    M = E-e*numpy.sin(E)
    return  M

# f=M2theta(3.6029,0.37255,1e-8) # From textbook - -8.6721

def gauss_eqn(t,yy,param):
    # Guass planetary equations
    # Input, 
    # t - time
    # x0 - state vector
    # param is a tuple 3 x 1
    # param[0] - COE vector - [angular momentum, eccentricity, inclination, RAAN, argument of perigee, true anomaly]
    # param[1] - J2 constant value
    # param[0] - list of information related to Earth [mu, radius]

    mu=param
    y_dot=numpy.zeros((6,))



    # assigning the state variables
    a = yy[0]
    e = yy[1]
    i = yy[2]
    OM = yy[3]
    om = yy[4]
    theta = yy[5]
    u=theta+om


    if theta>2*numpy.pi:
        theta=theta-2*numpy.pi

    h=numpy.sqrt(mu*a*(1-e**2))
    term1=(h**2)/(mu)
    p=term1
    rp=a*(1-e)
    r = p / ( 1 + e * numpy.cos( theta ) ) 
    n = numpy.sqrt(mu/(a**3))

    rr,vv=kep2car(numpy.array([h,yy[1],yy[2],yy[3],yy[4],yy[5]]),mu)
    # data={"J":[J2,J3,J4],"S/C":[M_SC,A_cross,C_D,Ballistic coefficient],"Primary":[mu,RE.w]}
    data={"J":[0.1082626925638815e-2,0,0],"S/C":[300,2,0.9,300],"Primary":[3.98600433e5,6378.16,7.2921150e-5]}

    # perturbations
    a_J=aspherical_perturbation(rr,data,1)
    a_drag=numpy.zeros(3) #atmosphheric_drag(rr,vv,data)
    # a_J = numpy.zeros(3)
    a_per = a_J 

    F_J=numpy.matmul(RSW2ECI(om,OM,i,theta),a_per)
    # Vallado page: 164 convcersion
    r_unit=rr / (rr[0]**2 + rr[1]**2 + rr[2]**2)**0.5
    w_unit=numpy.cross(rr,vv)/numpy.linalg.norm(numpy.cross(rr,vv))
    s_unit=numpy.cross(w_unit,r_unit)

    t_h = vv / numpy.linalg.norm( vv )                                          # velocity versoe
    h_h = numpy.cross( rr, vv ) / numpy.linalg.norm(numpy.cross(rr, vv ) )           # angular momentum veror
    n_h = numpy.cross( h_h, t_h ) 



    Rot_RSW=numpy.concatenate((r_unit, s_unit, w_unit)).reshape((-1, 3), order='F')
    Rot_RSW=numpy.concatenate((t_h, h_h, n_h)).reshape((-1, 3), order='F')
    F_J=numpy.matmul(Rot_RSW,a_per)
    #F_J=numpy.zeros(3)
    FR=F_J[0]
    FS=F_J[1]
    FW=F_J[2]

    
    y_dot[0]=(2/(n*numpy.sqrt(1-e**2)))*((e*numpy.sin(theta)*FR)+(p/r)*FS)

    y_dot[1]=((numpy.sqrt(1-e**2))/(n*a))*((numpy.sin(theta)*FR)+(numpy.cos(theta)+((e+numpy.cos(theta))/(1+e*numpy.cos(theta))))*FS)
    
    y_dot[2]=((r*numpy.cos(u))/(n*(a**2)*numpy.sqrt(1-e**2)))*FW
    
    y_dot[3]=((r*numpy.sin(u))/(n*(a**2)*numpy.sqrt(1-e**2)*numpy.sin(i)))*FW
    
    y_dot[4]=((numpy.sqrt(1-e**2))/(n*a*e))*((-numpy.cos(theta)*FR)+numpy.sin(theta)*(1+(r/p))*FS)-((r*(1/numpy.tan(i))*numpy.sin(u))/h)*FW

    y_dot[5]=(h/r**2)+(1/(e*h))*(p*numpy.cos(theta))*FR-(p+r)*numpy.sin(theta)*FS
    
    
    return y_dot

# guess event

def Event_COE(t,yy,param):
    # Guass planetary equations
    # Input, 
    # t - time
    # x0 - state vector
    # param is a tuple 3 x 1
    # param[0] - COE vector - [angular momentum, eccentricity, inclination, RAAN, argument of perigee, true anomaly]
    # param[1] - J2 constant value
    # param[0] - list of information related to Earth [mu, radius]

    mu=param
    y_dot=numpy.zeros((6,))



    # assigning the state variables
    a = yy[0]
    e = yy[1]
    i = yy[2]
    OM = yy[3]
    om = yy[4]
    theta = yy[5]
    u=theta+om


    if theta>2*numpy.pi:
        theta=theta-2*numpy.pi

    h=numpy.sqrt(mu*a*(1-e**2))

    rr,vv=kep2car(numpy.array([h,yy[1],yy[2],yy[3],yy[4],yy[5]]),mu)
    # data={"J":[J2,J3,J4],"S/C":[M_SC,A_cross,C_D,Ballistic coefficient],"Primary":[mu,RE.w]}
    data={"J":[0.1082626925638815e-2,0,0],"S/C":[300,2,0.9,300],"Primary":[3.98600433e5,6378.16,7.2921150e-5]}
    R_E=data["Primary"][1]

    r=(rr[0]**2 + rr[1]**2 + rr[2]**2)**0.5 # position magnitude

    if r-R_E < 100 :
       return 0
    else:
        return r 


def guess_nonsingular(t,yy,param):
    # Guass planetary equations
    # Input, 
    # t - time
    # x0 - state vector
    # param is a tuple 3 x 1
    # param[0] - COE vector - [angular momentum, eccentricity, inclination, RAAN, argument of perigee, true anomaly]
    # param[1] - J2 constant value
    # param[0] - list of information related to Earth [mu, radius]

    mu=param
    y_dot=numpy.zeros((6,))



    # assigning the state variables
    a = yy[0]
    l = yy[1]
    i = yy[2]
    q1 = yy[3]
    q2 = yy[4]
    OM = yy[5]



    e=numpy.sqrt(q1**2 + q2**2)
    h=numpy.sqrt(mu*a*(1-e**2))
    term1=(h**2)/(mu)
    neta = 1- q1**2 - q2**2
    p=term1
    rp=a*(1-e)
    r = ( a*neta**2 ) / (1+q1)
    n = numpy.sqrt(mu/(a**3))

    omega_peri = numpy.arcsin(q1 / e)
    mean_anamoly = l-omega_peri
    theta_tuple = M2theta(mean_anamoly,e,1e-8)
    theta =theta_tuple[0]
    u=theta+omega_peri

    # rr,vv=kep2car(numpy.array([h,yy[1],yy[2],yy[3],yy[4],yy[5]]),mu)
    # data={"J":[J2,J3,J4],"S/C":[M_SC,A_cross,C_D,Ballistic coefficient],"Primary":[mu,RE.w]}
    data={"J":[0.1082626925638815e-2,0,0],"S/C":[300,2,0.9,300],"Primary":[3.98600433e5,6378.16,7.2921150e-5]}

    # perturbations
    # a_J=aspherical_perturbation(rr,data,1)
    a_drag=numpy.zeros(3) #atmosphheric_drag(rr,vv,data)
    # a_J = numpy.zeros(3)
    # a_per = a_J 

    F_J = numpy.zeros(3)
    #F_J=numpy.matmul(RSW2ECI(om,OM,i,theta),a_per)
    # Vallado page: 164 convcersion
    # r_unit=rr / (rr[0]**2 + rr[1]**2 + rr[2]**2)**0.5
    # w_unit=numpy.cross(rr,vv)/numpy.linalg.norm(numpy.cross(rr,vv))
    # s_unit=numpy.cross(w_unit,r_unit)

    # t_h = vv / numpy.linalg.norm( vv )                                          # velocity versoe
    # h_h = numpy.cross( rr, vv ) / numpy.linalg.norm(numpy.cross(rr, vv ) )           # angular momentum veror
    # n_h = numpy.cross( h_h, t_h ) 



    # Rot_RSW=numpy.concatenate((r_unit, s_unit, w_unit)).reshape((-1, 3), order='F')
    # Rot_RSW=numpy.concatenate((t_h, h_h, n_h)).reshape((-1, 3), order='F')
    # F_J=numpy.matmul(Rot_RSW,a_per)
    #F_J=numpy.zeros(3)
    FR=F_J[0]
    FS=F_J[1]
    FW=F_J[2]

    
    y_dot[0]=((2*a**2) / h) * (((q1*numpy.sin(u))-q2*numpy.cos(u))*FR + (p/r)*FS)

    t1= ((-p/h*(1+neta))*(q1*numpy.cos(u)+q2*numpy.sin(u))-((2*neta*r)/h))
    t2=((p+r)/h*(1+neta))*(q1*numpy.sin(u)-q2*numpy.cos(u))
    t3=((r*numpy.sin(u)*numpy.cos(i))/(h*numpy.sin(i)))

    y_dot[1]=t1*FR + t2 * FS - t3 *FW

    y_dot[2]= ((r*numpy.cos(u))/h) * FW

    t1= ((p*numpy.sin(u))/h)
    t2=((p+r)/h)*(numpy.cos(u)+r*q1)
    t3=((r*q2*numpy.sin(u)*numpy.cos(i))/(h*numpy.sin(i)))

    y_dot[3]=t1 * FR + t2 * FS + t3 * FW

    t1= ((p*numpy.cos(u))/h)
    t2=((p+r)/h)*(numpy.cos(u)+r*q2)
    t3=((r*q1*numpy.sin(u)*numpy.cos(i))/(h*numpy.sin(i)))
    
    y_dot[4]=-t1 * FR + t2 * FS - t3 * FW

    y_dot[5]=((r*numpy.cos(u))/(h*numpy.sin(i))) * FW
    
    
    return y_dot

def guess_nonsingular_Bmat(t,yy,param):
   # Guass planetary equations
    # Input, 
    # t - time
    # x0 - state vector
    # param is a tuple 3 x 1
    # param[0] - COE vector - [angular momentum, eccentricity, inclination, RAAN, argument of perigee, true anomaly]
    # param[1] - J2 constant value
    # param[0] - list of information related to Earth [mu, radius]

    mu=param
    y_dot=numpy.zeros((6,))



    # assigning the state variables
    a = yy[0]
    l = yy[1]
    i = yy[2]
    q1 = yy[3]
    q2 = yy[4]
    OM = yy[5]



    e=numpy.sqrt(q1**2 + q2**2)
    h=numpy.sqrt(mu*a*(1-e**2))
    term1=(h**2)/(mu)
    neta = 1- q1**2 - q2**2
    p=term1
    rp=a*(1-e)
    r = ( a*neta**2 ) / (1+q1)
    n = numpy.sqrt(mu/(a**3))

    omega_peri = numpy.arcsin(q1 / e)
    mean_anamoly = l-omega_peri
    theta_tuple = M2theta(mean_anamoly,e,1e-8)
    theta =theta_tuple[0]
    u=theta+omega_peri

    # rr,vv=kep2car(numpy.array([h,yy[1],yy[2],yy[3],yy[4],yy[5]]),mu)
    # data={"J":[J2,J3,J4],"S/C":[M_SC,A_cross,C_D,Ballistic coefficient],"Primary":[mu,RE.w]}
    data={"J":[0.1082626925638815e-2,0,0],"S/C":[300,2,0.9,300],"Primary":[3.98600433e5,6378.16,7.2921150e-5]}

    # perturbations
    # a_J=aspherical_perturbation(rr,data,1)
    a_drag=numpy.zeros(3) #atmosphheric_drag(rr,vv,data)
    # a_J = numpy.zeros(3)
    # a_per = a_J 

    F_J = numpy.zeros(3)
    #F_J=numpy.matmul(RSW2ECI(om,OM,i,theta),a_per)
    # Vallado page: 164 convcersion
    # r_unit=rr / (rr[0]**2 + rr[1]**2 + rr[2]**2)**0.5
    # w_unit=numpy.cross(rr,vv)/numpy.linalg.norm(numpy.cross(rr,vv))
    # s_unit=numpy.cross(w_unit,r_unit)

    # t_h = vv / numpy.linalg.norm( vv )                                          # velocity versoe
    # h_h = numpy.cross( rr, vv ) / numpy.linalg.norm(numpy.cross(rr, vv ) )           # angular momentum veror
    # n_h = numpy.cross( h_h, t_h ) 



    # Rot_RSW=numpy.concatenate((r_unit, s_unit, w_unit)).reshape((-1, 3), order='F')
    # Rot_RSW=numpy.concatenate((t_h, h_h, n_h)).reshape((-1, 3), order='F')
    # F_J=numpy.matmul(Rot_RSW,a_per)
    #F_J=numpy.zeros(3)
    FR=F_J[0]
    FS=F_J[1]
    FW=F_J[2]


    
    y_dot_0=numpy.array([((2*a**2) / h) * (((q1*numpy.sin(u))-q2*numpy.cos*(u)), +(p/r)*FS),0])

    t1= ((-p/h*(1+neta))*(q1*numpy.cos(u)+q2*numpy.sin(u))-((2*neta*r)/h))
    t2=((p+r)/h*(1+neta))*(q1*numpy.sin(u)-q2*numpy.cos(u))
    t3=((r*numpy.sin(u)*numpy.cos(i))/(h*numpy.sin(i)))
    y_dot_1=numpy.array([t1,+t2,-t3 ])

    y_dot_2=numpy.array([0,0,((r*numpy.cos(u))/h)])

    t1= ((p*numpy.sin(u))/h)
    t2=((p+r)/h)*(numpy.cos(u)+r*q1)
    t3=((r*q2*numpy.sin(u)*numpy.cos(i))/(h*numpy.sin(i)))

    y_dot_3=numpy.array([t1,t2,+t3])

    t1= ((p*numpy.cos(u))/h)
    t2=((p+r)/h)*(numpy.cos(u)+r*q2)
    t3=((r*q1*numpy.sin(u)*numpy.cos(i))/(h*numpy.sin(i)))
    
    y_dot_4=numpy.array([-t1 + t2 ,- t3 ])

    y_dot_5=numpy.array([0,0,((r*numpy.cos(u))/(h*numpy.sin(i)))])
    
    B_mat=numpy.vstack([y_dot_0,y_dot_1,y_dot_2,y_dot_3,y_dot_4,y_dot_5])
    
    return B_mat

# def lagrage_J2_diff():
#     return 0


# # References frames and convertions

def Lagrange_deri(t,yy,param,q1_0,q2_0,t0,t):
    # Guass planetary equations
    # Input, 
    # t - time
    # x0 - state vector
    # param is a tuple 3 x 1
    # param[0] - COE vector - [angular momentum, eccentricity, inclination, RAAN, argument of perigee, true anomaly]
    # param[1] - J2 constant value
    # param[0] - list of information related to Earth [mu, radius]

    # data={"J":[J2,J3,J4],"S/C":[M_SC,A_cross,C_D,Ballistic coefficient],"Primary":[mu,RE.w]}
    data={"J":[0.1082626925638815e-2,0,0],"S/C":[300,2,0.9,300],"Primary":[3.98600433e5,6378.16,7.2921150e-5]}
    mu=param
    y_dot=numpy.zeros((6,))
    epsilon =  data["J"][0]



    # assigning the state variables
    a = yy[0]
    l = yy[1]
    i = yy[2]
    q1 = yy[3]
    q2 = yy[4]
    OM = yy[5]



    e=numpy.sqrt(q1**2 + q2**2)
    h=numpy.sqrt(mu*a*(1-e**2))
    term1=(h**2)/(mu)
    neta = 1- q1**2 - q2**2
    p=term1
    rp=a*(1-e)
    r = ( a*neta**2 ) / (1+q1)
    n = numpy.sqrt(mu/(a**3))

    omega_peri = numpy.arcsin(q1 / e)
    mean_anamoly = l-omega_peri
    theta_tuple = M2theta(mean_anamoly,e,1e-8)
    theta =theta_tuple[0]
    u=theta+omega_peri

    # rr,vv=kep2car(numpy.array([h,yy[1],yy[2],yy[3],yy[4],yy[5]]),mu)



    # perturbations
    # a_J=aspherical_perturbation(rr,data,1)
    a_drag=numpy.zeros(3) #atmosphheric_drag(rr,vv,data)
    # a_J = numpy.zeros(3)
    # a_per = a_J 

    F_J = numpy.zeros(3)
    #F_J=numpy.matmul(RSW2ECI(om,OM,i,theta),a_per)
    # Vallado page: 164 convcersion
    # r_unit=rr / (rr[0]**2 + rr[1]**2 + rr[2]**2)**0.5
    # w_unit=numpy.cross(rr,vv)/numpy.linalg.norm(numpy.cross(rr,vv))
    # s_unit=numpy.cross(w_unit,r_unit)

    # t_h = vv / numpy.linalg.norm( vv )                                          # velocity versoe
    # h_h = numpy.cross( rr, vv ) / numpy.linalg.norm(numpy.cross(rr, vv ) )           # angular momentum veror
    # n_h = numpy.cross( h_h, t_h ) 



    # Rot_RSW=numpy.concatenate((r_unit, s_unit, w_unit)).reshape((-1, 3), order='F')
    # Rot_RSW=numpy.concatenate((t_h, h_h, n_h)).reshape((-1, 3), order='F')
    # F_J=numpy.matmul(Rot_RSW,a_per)
    #F_J=numpy.zeros(3)
    FR=F_J[0]
    FS=F_J[1]
    FW=F_J[2]

    w_dot = ((3*epsilon)/4)* (((5*numpy.cos(i)**2)-1)) 
    q1 = q1_0 * numpy.cos(w_dot*(t-t0)) - q2_0 *numpy.sin(w_dot*(t-t0))
    q2 = q1_0 * numpy.sin(w_dot*(t-t0)) + q2_0 *numpy.cos(w_dot*(t-t0))


    
    term_l_a_1= (-(3*n)/(2*a)) 
    term_l_a_2= ((21*epsilon)/(8*a)) * (neta*((3*numpy.cos(i)**2)-1)+((5*numpy.cos(i)**2)-1)) 
    term_l_a= term_l_a_1 - term_l_a_2 

    term_l_i= ((-3*epsilon)/4)(3*neta+5)*numpy.sin(2*i)

    term_l_q1_1 = ((3*epsilon)/(4*neta**2)) * (3*neta*((3*numpy.cos(i)**2)-1)+4*((5*numpy.cos(i)**2)-1)) 
    term_l_q1 = term_l_q1_1   

    term_l_q2_1 = ((3*epsilon)/(4*neta**2)) * (3*neta*((3*numpy.cos(i)**2)-1)+4*((5*numpy.cos(i)**2)-1)) 
    term_l_q2 = term_l_q2_1 * q2  

    term_q1_a_1 = ((21*epsilon)/(8*a)) * (((5*numpy.cos(i)**2)-1)) 
    term_q1_a = term_q1_a * q2

    term_q1_i = ((15*epsilon)/(4)) * q2* (numpy.sin(2*i)) 

    term_q1_q1_1 = ((-3*epsilon)/(neta**2)) * (((5*numpy.cos(i)**2)-1)) 
    term_q1_q1 = term_q1_q1_1 * q1* q2

    term_q1_q2 = ((-3*epsilon)/(4)) *(1+((4*q2**2)/neta**2))* (((5*numpy.cos(i)**2)-1)) 

    term_q2_a_1= ((-21*epsilon)/(8*a)) * (((5*numpy.cos(i)**2)-1)) 
    term_q2_a = term_q2_a_1 * q1

    term_q2_i= ((-15*epsilon)/(4)) *  q1 *numpy.sin(2*i) 

    term_q2_q1 = ((3*epsilon)/(4)) *(1+((4*q2**2)/neta**2))* (((5*numpy.cos(i)**2)-1)) 

    term_q2_q2_1 = ((3*epsilon)/(neta**2)) * (((5*numpy.cos(i)**2)-1)) 
    term_q2_q2 = term_q1_q2 * q1 * q2

    term_OM_a= ((21*epsilon)/(4*a)) *  numpy.cos(i) 

    term_OM_i= ((3*epsilon)/(2)) *numpy.sin(i) 

    term_OM_q1= ((-6*epsilon)/(neta**2)) *  q1 *numpy.cos(i) 

    term_OM_q2= ((-6*epsilon)/(neta**2)) *  q2 *numpy.cos(i) 

    A_mat = numpy.vstack([[0,0,0,0,0,0],
                         [term_l_a,0,term_l_i, term_l_q1, term_l_q2,0],
                         [0,0,0,0,0,0],
                         [term_q1_a,0,term_q1_i,term_q1_q1,term_q1_q2,0],
                         [term_q2_a,0,term_q2_i,term_q2_q1,term_q2_q2,0],
                         [term_OM_a,0,term_OM_i,term_OM_q1,term_OM_q2,0]])
        
    return y_dot



def Cart2RO(RO,OE_1):
    # Guass planetary equations
    # Input, 
    # t - time
    # x0 - state vector
    # param is a tuple 3 x 1
    # param[0] - COE vector - [angular momentum, eccentricity, inclination, RAAN, argument of perigee, true anomaly]
    # param[1] - J2 constant value
    # param[0] - list of information related to Earth [mu, radius]

    # data={"J":[J2,J3,J4],"S/C":[M_SC,A_cross,C_D,Ballistic coefficient],"Primary":[mu,RE.w]}
    data={"J":[0.1082626925638815e-2,0,0],"S/C":[300,2,0.9,300],"Primary":[3.98600433e5,6378.16,7.2921150e-5]}
    mu=param
    y_dot=numpy.zeros((6,))
    epsilon =  data["J"][0]



    # assigning the state variables
    a =OE_1[0]
    l =OE_1[1]
    i =OE_1[2]
    q1 =OE_1[3]
    q2 =OE_1[4]
    OM =OE_1[5]



    e=numpy.sqrt(q1**2 + q2**2)
    h=numpy.sqrt(mu*a*(1-e**2))
    term1=(h**2)/(mu)
    neta = 1- q1**2 - q2**2
    p=term1
    rp=a*(1-e)
    r = ( a*neta**2 ) / (1+q1)
    n = numpy.sqrt(mu/(a**3))

    omega_peri = numpy.arcsin(q1 / e)
    mean_anamoly = l-omega_peri
    theta_tuple = M2theta(mean_anamoly,e,1e-8)
    theta =theta_tuple[0]
    u=theta+omega_peri

    Vr = (h/p) * (q1*numpy.sin(u)-q2*numpy.cos(u))
    Vt = (h/p) * (1+q1*numpy.cos(u)+q2*numpy.sin(u))
    
    x = (r/a)* RO[0]+ (Vr/Vt)*r*

    return y_dot

# create a simple function to do quick sort of the array and return the index of the sorted array   