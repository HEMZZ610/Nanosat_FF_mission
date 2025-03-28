"""
Nanosat Formation Flying Project

All the functions related to the orbital mechanics are defined here

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

def NSROE2car(ROE,param):
    # Guass planetary equations
    # Input, 
    # t - time
    # x0 - state vector
    # param is a tuple 3 x 1
    # param[0] - COE vector - [angular momentum, eccentricity, inclination, RAAN, argument of perigee, true anomaly]
    # param[1] - J2 constant value
    # param[0] - list of information related to Earth [mu, radius]

    mu=param["Primary"][0]
    y_dot=numpy.zeros((6,))



    # assigning the state variables
    a = ROE[0]
    l = ROE[1]
    i = ROE[2]
    q1 = ROE[3]
    q2 = ROE[4]
    OM = ROE[5]



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
    
    # obtaining position and velocity vector in perifocan reference frame
    rp = ( h ** 2 / mu ) * ( 1 / ( 1 + e * numpy.cos( u ) ) ) * ( numpy.cos( u ) * numpy.array([ 1 , 0 ,0 ])
        + numpy.sin( u ) * numpy.array([ 0, 1, 0 ]) )
    vp = ( mu / h ) * ( -numpy.sin( u ) * numpy.array([ 1 , 0 , 0 ]) + ( e + numpy.cos( u ) ) * numpy.array([ 0 , 1 , 0 ]) ) ;


    RR=numpy.matmul(PQW2ECI(OM,omega_peri,i),numpy.transpose(rp))
    VV=numpy.matmul(PQW2ECI(OM,omega_peri,i),numpy.transpose(vp))

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
    a_J = numpy.zeros(3)
    a_per = a_J 

    #F_J=numpy.matmul(RSW2ECI(om,OM,i,theta),a_per)
    # Vallado page: 164 convcersion -RSW frame
    r_unit=rr / (rr[0]**2 + rr[1]**2 + rr[2]**2)**0.5
    w_unit=numpy.cross(rr,vv)/numpy.linalg.norm(numpy.cross(rr,vv))
    s_unit=numpy.cross(w_unit,r_unit)

    #print("\n",numpy.linalg.norm(r_unit),"|",numpy.linalg.norm(w_unit),"|",numpy.linalg.norm(s_unit))
    # t_h = vv / numpy.linalg.norm( vv )                                          # velocity versoe
    # h_h = numpy.cross( rr, vv ) / numpy.linalg.norm(numpy.cross(rr, vv ) )           # angular momentum veror
    # n_h = numpy.cross( h_h, t_h ) 


    # matrix to convert from IJK frame to RSW frame
    Rot_RSW = numpy.vstack([r_unit, s_unit, w_unit])



    # Convert J2 perturbation from RSW to IJK frame
    F_J= numpy.matmul(Rot_RSW, a_J)


    #Rot_RSW=numpy.concatenate((t_h, h_h, n_h)).reshape((-1, 3), order='F')
    #F_J=numpy.matmul(Rot_RSW_temp,a_per)
    #F_J=numpy.zeros(3)
    FR=F_J[0]
    FS=F_J[1]
    FW=F_J[2]

    # print(FR,"|",FS,"|",F_J)

    
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

    t1= ((-p/h*(1+eta))*(q1*numpy.cos(u)+q2*numpy.sin(u))-((2*eta*r)/h))
    t2=((p+r)/h*(1+eta))*(q1*numpy.sin(u)-q2*numpy.cos(u))
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

    mu=param["Primary"][0]
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
    eta =  numpy.sqrt(1 - q1**2 - q2**2)
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

    y_dot_0 = ((2 * a**2) / h)* numpy.array([((q1 * numpy.sin(u)) - q2 * numpy.cos(u)), (p / r) , 0])

    t1 = (-p / (h * (1 + eta))) * (q1 * numpy.cos(u) + q2 * numpy.sin(u)) - ((2 * eta * r) / h)
    t2 = ((p + r) / (h * (1 + eta))) * (q1 * numpy.sin(u) - q2 * numpy.cos(u))
    t3 = (r * numpy.sin(u) * numpy.cos(i)) / (h * numpy.sin(i))

    y_dot_1 = numpy.array([t1, t2, -t3])

    y_dot_2 = numpy.array([0, 0, (r * numpy.cos(u)) / h])

    t1 = (p * numpy.sin(u)) / h
    t2 = ((p + r) / h) * (numpy.cos(u) + r * q1)
    t3 = (r * q2 * numpy.sin(u) * numpy.cos(i)) / (h * numpy.sin(i))
    y_dot_3 = numpy.array([t1, t2, t3])

    t1 = (p * numpy.cos(u)) / h
    t2 = ((p + r) / h) * (numpy.sin(u) + r * q2)
    t3 = (r * q1 * numpy.sin(u) * numpy.cos(i)) / (h * numpy.sin(i))
    y_dot_4 = numpy.array([-t1, t2, -t3])

    y_dot_5 = numpy.array([0, 0, (r * numpy.sin(u)) / (h * numpy.sin(i))])

    B_mat = numpy.vstack([y_dot_0, y_dot_1, y_dot_2, y_dot_3, y_dot_4, y_dot_5])
        
    return B_mat

def lagrage_J2_diff(t,yy,data):
    
    f_dot=numpy.zeros(6)
    # data={"J":[J2,J3,J4],"S/C":[M_SC,A_cross,C_D,Ballistic coefficient],"Primary":[mu,RE.w]}
    data={"J":[0.1082626925638815e-2,0,0],"S/C":[300,2,0.9,300],"Primary":[3.98600433e5,6378.16,7.2921150e-5]}
    mu=data["Primary"][0]
    Re=data["Primary"][1]
    y_dot=numpy.zeros((6,))
    J2 =  data["J"][0]



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
    eta =  numpy.sqrt(1 - q1**2 - q2**2)
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



    # Compute each component
    component_1 = 0
    component_2 = n + (3/4) * J2 * (Re / p)**2 * n * (eta * (3 * numpy.cos(i)**2 - 1) + (5 * numpy.cos(i)**2 - 1))
    component_3 = 0
    component_4 = - (3/4) * J2 * (Re / p)**2 * n * (3 * numpy.cos(i)**2 - 1) * q2
    component_5 = (3/4) * J2 * (Re / p)**2 * n * (3 * numpy.cos(i)**2 - 1) * q1
    component_6 = - (3/2) * J2 * (Re / p)**2 * n * numpy.cos(i)
    
    # Combine components into a vector
    f_dot = numpy.array([component_1, component_2, component_3, component_4, component_5, component_6])
    
    return f_dot
#     return 0




# # References frames and convertions

def Lagrange_deri(t,yy,param):
    # taken from ROSCOE ET AL Appendix B Differential Form of Lagrange’s Planetary Equations

    # data={"J":[J2,J3,J4],"S/C":[M_SC,A_cross,C_D,Ballistic coefficient],"Primary":[mu,RE.w]}
    data=param# {"J":[0.1082626925638815e-2,0,0],"S/C":[300,2,0.9,300],"Primary":[3.98600433e5,6378.16,7.2921150e-5]}
    mu=data["Primary"][0]
    y_dot=numpy.zeros((6,))
    



    # assigning the state variables
    a = yy[0]
    l = yy[1]
    i = yy[2]
    q1 = yy[3]
    q2 = yy[4]
    OM = yy[5]

    q1_0 = param["Init"][0]
    q2_0 = param["Init"][1]
    t0 = param["Init"][2]



    e=numpy.sqrt(q1**2 + q2**2)
    h=numpy.sqrt(mu*a*(1-e**2))
    term1=(h**2)/(mu)
    eta =  numpy.sqrt(1 - q1**2 - q2**2)
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
        
    # rr,vv=kep2car(numpy.array([h,yy[1],yy[2],yy[3],yy[4],yy[5]]),mu)

    epsilon =  data["J"][0] * ((data["Primary"][1] / p)**2) * n


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
    term_l_a_2= ((21*epsilon)/(8*a)) * (eta*((3*numpy.cos(i)**2)-1)+((5*numpy.cos(i)**2)-1)) 
    term_l_a= term_l_a_1 - term_l_a_2 

    term_l_i= ((-3*epsilon)/4)*(3*eta+5)*numpy.sin(2*i)

    term_l_q1_1 = ((3*epsilon)/(4*eta**2)) * (3*eta*((3*numpy.cos(i)**2)-1)+4*((5*numpy.cos(i)**2)-1)) 
    term_l_q1 = term_l_q1_1 * q1

    term_l_q2_1 = ((3*epsilon)/(4*eta**2)) * (3*eta*((3*numpy.cos(i)**2)-1)+4*((5*numpy.cos(i)**2)-1)) 
    term_l_q2 = term_l_q2_1 * q2  

    term_q1_a_1 = ((21*epsilon)/(8*a)) * (((5*numpy.cos(i)**2)-1)) 
    term_q1_a = term_q1_a_1 * q2

    term_q1_i = ((15*epsilon)/(4)) * q2* (numpy.sin(2*i)) 

    term_q1_q1_1 = ((-3*epsilon)/(eta**2)) * (((5*numpy.cos(i)**2)-1)) 
    term_q1_q1 = term_q1_q1_1 * q1* q2

    term_q1_q2 = ((-3*epsilon)/(4)) *(1+((4*q2**2)/eta**2))* (((5*numpy.cos(i)**2)-1)) 

    term_q2_a_1= ((-21*epsilon)/(8*a)) * (((5*numpy.cos(i)**2)-1)) 
    term_q2_a = term_q2_a_1 * q1

    term_q2_i= ((-15*epsilon)/(4)) *  q1 *numpy.sin(2*i) 

    term_q2_q1 = ((3*epsilon)/(4)) *(1+((4*q2**2)/eta**2))* (((5*numpy.cos(i)**2)-1)) 

    term_q2_q2_1 = ((3*epsilon)/(eta**2)) * (((5*numpy.cos(i)**2)-1)) 
    term_q2_q2 = term_q2_q2_1 * q1 * q2

    term_OM_a= ((21*epsilon)/(4*a)) *  numpy.cos(i) 

    term_OM_i= ((3*epsilon)/(2)) *numpy.sin(i) 

    term_OM_q1= ((-6*epsilon)/(eta**2)) *  q1 *numpy.cos(i) 

    term_OM_q2= ((-6*epsilon)/(eta**2)) *  q2 *numpy.cos(i) 

    A_mat = numpy.vstack([[0,0,0,0,0,0],
                         [term_l_a,0,term_l_i, term_l_q1, term_l_q2,0],
                         [0,0,0,0,0,0],
                         [term_q1_a,0,term_q1_i,term_q1_q1,term_q1_q2,0],
                         [term_q2_a,0,term_q2_i,term_q2_q1,term_q2_q2,0],
                         [term_OM_a,0,term_OM_i,term_OM_q1,term_OM_q2,0]])
        
    return A_mat

def yaw_dynamics(t,yy,param):

    Izc=param["sat"][0]
    Izd=param["sat"][1]

    y_dot=numpy.zeros((2,))

    y_dot[0]=-Izc*yy[0]
    y_dot[1]=-Izd*yy[1]

    return y_dot

def Dynamics(t,yy,param):


    y_dot_chief=absolute_NSROE_dynamics(t,yy[6:12],param)
    
    A=Lagrange_deri(t,yy[6:12],param)
    B=guess_nonsingular_Bmat(t,yy[6:12],param)
    
    y_dot_yaw=yaw_dynamics(t,yy[12:14],param)
    
    y_dot=numpy.matmul(A,yy[0:6])+numpy.matmul(B,numpy.array([0,0,0]))

    y = numpy.concatenate((y_dot,y_dot_chief,y_dot_yaw))

    return y

def absolute_NSROE_dynamics(t,yy,param):
    
    A=lagrage_J2_diff(t,yy,param)
    B=guess_nonsingular_Bmat(t,yy,param)
    y_dot=A+numpy.matmul(B,numpy.array([0,0,0]))

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
    mu=data["Primary"][0]
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

    Vr = (h/p) * (q1*numpy.sin(u)-q2*numpy.cos(u))
    Vt = (h/p) * (1+q1*numpy.cos(u)+q2*numpy.sin(u))
    
    #x = (r/a)* RO[0]+ (Vr/Vt)*r*

    return y_dot



# converts the design parameters to relative orbital elements


def Param2NROE(NOE, parameters,data):

    # Taken from  C.traub paper 
    # NOE=numpy.array([a,lambda_0,i,q1,q2,omega]) unpack this
    a, lambda_, i, q1, q2, omega = NOE
    
    # Unpacking the parameters
    rho_1, rho_2, rho_3, alpha_0, beta_0,v_d = parameters
    
    mu=data["Primary"][0]

    eta = numpy.sqrt(1 - q1**2 - q2**2)

    n = numpy.sqrt(mu / a**3)

    e=numpy.sqrt(q1**2 + q2**2)
    h=numpy.sqrt(mu*a*(1-e**2))
    term1=(h**2)/(mu)
    p=term1

    delta_a = (-2 * eta * v_d) / (3 * n)
    
    # Equation 16 (uses 'omega' from orbital elements)
    delta_Omega = (-rho_3 * numpy.sin(beta_0)) / (p * numpy.sin(i))
    
    # Equation 12 (uses 'lambda_' from orbital elements)
    delta_lambda = (rho_2 / p) - delta_Omega * numpy.cos(i) - ((1 + eta + eta**2) / (1 + eta)) * (rho_1 / p) * (q1 * numpy.cos(alpha_0) - q2 * numpy.sin(alpha_0))
    
    # Equation 13
    delta_i = (rho_3 / p) * numpy.cos(beta_0)
    
    # Equation 14
    delta_q1 = -(1 - q1**2) * (rho_1 / p) * numpy.sin(alpha_0) + q1 * q2 * (rho_1 / p) * numpy.cos(alpha_0) - q2 * (rho_2 / p - delta_Omega * numpy.cos(i))
    
    # Equation 15
    delta_q2 = -(1 - q2**2) * (rho_1 / p) * numpy.cos(alpha_0) + q1 * q2 * (rho_1 / p) * numpy.sin(alpha_0) + q1 * (rho_2 / p - delta_Omega * numpy.cos(i))
    
    # Return as vector
    return numpy.array([delta_a, delta_lambda, delta_i, delta_q1, delta_q2, delta_Omega])



# Convert from NSROE to Cartesian

def NSROE2Cart(NSROE,NSROE0,x_vec_init,data):
    
    a, lambda_, i, q1, q2, omega =NSROE 
    a0, lambda_0, i0, q1_0, q2_0, omega_0 =NSROE0
    x_0, y_0, z_0, x_dot_0, y_dot_0, z_dot_0 = x_vec_init
    
    mu = data["Primary"][0]

    # initial parameters
    e0=numpy.sqrt(q1_0**2 + q2_0**2)
    h0=numpy.sqrt(mu*a0*(1-e0**2))
    term1_0=(h0**2)/(mu)
    eta0 = 1- q1_0**2 - q2_0**2
    p0=term1_0
    rp0=a0*(1-e0)
    r = ( a0*eta0**2 ) / (1+q1_0)
    n0 = numpy.sqrt(mu/(a0**3))

    omega_peri0 = numpy.arcsin(q1_0 / e0)
    mean_anamoly0 = lambda_0-omega_peri0
    theta_tuple0 = M2theta(mean_anamoly0,e0,1e-8)
    f0 =theta_tuple0[0]
    theta_0=f0+omega_peri0
    E0 = 2*numpy.arctan(numpy.tan(f0/2)*numpy.sqrt((1-e0)/(1+e0)))
    F0 = omega_peri0 + E0 

    # current parameters
    e=numpy.sqrt(q1**2 + q2**2)
    h=numpy.sqrt(mu*a*(1-e**2))
    term1=(h**2)/(mu)
    eta =  numpy.sqrt(1 - q1**2 - q2**2)
    p=term1
    rp=a*(1-e)
    n = numpy.sqrt(mu/(a**3))

    omega_peri = numpy.arccos(q1 / e)
    mean_anamoly = lambda_-omega_peri
    theta_tuple = M2theta(mean_anamoly,e,1e-8)
    f =theta_tuple[0]
    theta=f+omega_peri
    E = 2*numpy.arctan(numpy.tan(f/2)*numpy.sqrt((1-e)/(1+e)))
    F = omega_peri + E 
    r = ( a*eta**2 ) / (1+ (q1 * numpy.cos(F)) + (q2 * numpy.sin(F)))


    # Calculate alpha
    alpha_val = lambda q1,q2,theta: 1 + q1 * numpy.cos(theta) + q2 * numpy.cos(theta)
    
    # Calculate beta
    beta_val =  lambda q1,q2,theta: q1 * numpy.sin(theta) - q2 * numpy.cos(theta)
    
    # Calculate K(θ)
    K_val = lambda q1,q2,theta,F,F0 : (F - q1 * numpy.sin(F) + q2 * numpy.cos(F)) - (F0 - q1 * numpy.sin(F0) + q2 * numpy.cos(F0)) #  (lambda_ - lambda0)
    
    e = numpy.sqrt(q1**2 + q2**2)


    # Calculate c1
    c1 = - (3 / ((eta **2 )*  alpha_val(q1,q2,theta_0))) * \
         (q1*(1 + numpy.cos(theta_0)**2) + q2*numpy.cos(theta_0)*numpy.sin(theta_0) + \
         (2 - eta**2) * numpy.cos(theta_0)) * x_0 - \
         (1/eta**2) *(-q2*(1 + numpy.cos(theta_0)**2) + q1*numpy.cos(theta_0)*numpy.sin(theta_0)+numpy.sin(theta_0))*x_dot_0 - \
         (1/eta**2) *(q1*(1 + numpy.cos(theta_0)**2) + q2*numpy.cos(theta_0)*numpy.sin(theta_0)+2*numpy.cos(theta_0))*y_dot_0

    # Calculate c2 
    c2 = - (3 / ((eta **2 )*  alpha_val(q1,q2,theta_0))) * \
         (q2*(1 + numpy.sin(theta_0)**2) + q1*numpy.cos(theta_0)*numpy.sin(theta_0) + \
         (2 - eta**2) * numpy.sin(theta_0)) * x_0 - \
         (1/eta**2) *(q1*(1 + numpy.sin(theta_0)**2) - q2*numpy.cos(theta_0)*numpy.sin(theta_0)-numpy.cos(theta_0))*x_dot_0 - \
         (1/eta**2) *(q2*(1 + numpy.sin(theta_0)**2) + q1*numpy.cos(theta_0)*numpy.sin(theta_0)+2*numpy.sin(theta_0))*y_dot_0

    # Calculate c3
    c3 = (2 + 3 * e * numpy.cos(theta_0) + e**2) * x_0 + \
         e * numpy.sin(theta_0) * (1 + e * numpy.cos(theta_0)) * x_dot_0 + \
         (1 + e * numpy.cos(theta_0))**2 * y_dot_0

    # Calculate c4
    c4 = -(1/eta**2)*(1+alpha_val(q1,q2,theta_0)) * \
                    (((3*beta_val(q1,q2,theta_0)/alpha_val(q1,q2,theta_0))) *x_0 + \
                    (2-alpha_val(q1,q2,theta_0)) * x_dot_0 + beta_val(q1,q2,theta_0) * y_dot_0) + y_0
                    

    c5 = numpy.cos(theta_0)* z_0 - numpy.sin(theta_0) * z_dot_0

    c6 = numpy.sin(theta_0)* z_0 + numpy.cos(theta_0) * z_dot_0
    
    # Calculate x(θ)
    x_theta_val = (c1 * numpy.cos(theta) + c2 * numpy.sin(theta)) * alpha_val(q1,q2,theta) + \
                  (2 * c3 / eta**2) * (1 - (3 / (2 * eta**2)) * beta_val(q1,q2,theta) * alpha_val(q1,q2,theta) * K_val(q1,q2,theta,F,F0))
    
    # Calculate y(θ)
    y_theta_val = (-c1 * numpy.sin(theta) + c2 * numpy.cos(theta)) * (1+alpha_val(q1,q2,theta)) - \
                  (3 * c3 / eta**2) * (alpha_val(q1,q2,theta)**2) * K_val(q1,q2,theta,F,F0) + c4
    
    # Calculate z(θ)
    z_theta_val = c3 * numpy.cos(theta) + c6 * numpy.sin(theta)

    
    # Return x(θ), y(θ), z(θ) as a numpy vector
    return numpy.array([x_theta_val, y_theta_val, z_theta_val])



def NSROE2LVLH(NSROE,NSOE0,data):

    # data={"J":[J2,J3,J4],"S/C":[M_SC,A_cross,C_D,Ballistic coefficient],"Primary":[mu,RE.w]}
    data={"J":[0.1082626925638815e-2,0,0],"S/C":[300,2,0.9,300],"Primary":[3.98600433e5,6378.16,7.2921150e-5]}
    mu=data["Primary"][0]
    Re=data["Primary"][1]
    y_dot=numpy.zeros((6,))
    J2 =  data["J"][0]



    # assigning the state variables
    a = NSOE0[0]
    l = NSOE0[1]
    i = NSOE0[2]
    q1 = NSOE0[3]
    q2 = NSOE0[4]
    OM = NSOE0[5]



    delta_a = NSROE[0]
    delta_lambda0 = NSROE[1]
    delta_i = NSROE [2]
    delta_q1 = NSROE[3]
    delta_q2 = NSROE[4]
    delta_Omega = NSROE[5]


    e=numpy.sqrt(q1**2 + q2**2)
    h=numpy.sqrt(mu*a*(1-e**2))
    term1=(h**2)/(mu)
    eta =  numpy.sqrt(1 - q1**2 - q2**2)
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


    e1 =(a / eta) * ((1 - eta**2) * delta_lambda0**2 + 2 * (q2 * delta_q1 - q1 * delta_q2) * delta_lambda0
        - (q1 * delta_q1 + q2 * delta_q2)**2 + delta_q1**2 + delta_q2**2)**0.5


    e2 = p * (delta_Omega * numpy.cos(i) + ( (1 + eta + eta**2) / (eta**3 * (1 + eta)) ) * (q2 * delta_q1 - q1 * delta_q2) 
        + (1 / eta**3) * delta_lambda0 )


    e3 =   p * (delta_i**2 + delta_Omega**2 * numpy.sin(i)**2)**0.5


    alpha_numerator = (1 + eta) * (delta_q1 + q2 * delta_lambda0) - q1 * (q1 * delta_q1 + q2 * delta_q2)
    
    alpha_denominator = (1 + eta) * (delta_q2 - q1 * delta_lambda0) - q2 * (q1 * delta_q1 + q2 * delta_q2)
    
    alpha_0 = numpy.arctan2(  alpha_numerator, alpha_denominator)

    
    beta_0 = numpy.arctan2( - delta_Omega * numpy.sin(i) , delta_i)
    
    
    x_p = (e1 / p) * numpy.sin(u + alpha_0) * (1 + q1 * numpy.cos(u) + q2 * numpy.sin(u))

    
    y_p = (e1 / p) * numpy.cos(u + alpha_0) * (2 + q1 * numpy.cos(u) + q2 * numpy.sin(u)) + (e2 / p)

    
    z_p = (e3 / p) * numpy.sin(u + beta_0)

    return numpy.array([x_p, y_p, z_p])
