"""
Usage:
    

Author:
    Vishnuvardhan Shakthibala
"""

import numpy
import math

## Directional cosine matrix
# Taken from Bong Wie Space vehicle dynamics and control

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


## Euler angles
# First rotation is about arbitrary axis
# second rotation is about one of the other two axis.
# Third rotation is about axis that is not used for second rotation.
# 12 combinations

def Euler_rotation(sequence,theta):
    
    if sequence[0]==1:
        C1=C_1(theta[0])
    elif sequence[0]==2:
        C1=C_2(theta[0])
    elif sequence[0]==3:
        C1=C_3(theta[0])
     
    if sequence[1]==1:
        C2=C_1(theta[0])
    elif sequence[1]==2:
        C2=C_2(theta[0])
    elif sequence[1]==3:
        C2=C_3(theta[0])

    if sequence[2]==1:
        C3=C_1(theta[0])
    elif sequence[2]==2:
        C3=C_2(theta[0])
    elif sequence[2]==3:
        C3=C_3(theta[0])
    
    C=numpy.matmul(C1,numpy.matmul(C2,C3))
    
    return C

## Quaternions to DCM
def Q2DCM(q_vector):
    q1=q_vector[0]
    q2=q_vector[1]
    q3=q_vector[2]
    q4=q_vector[3]
    C=numpy.array([1-2*((q2**2)+q3**2),2*(q2*q3+q3*q4),2*(q1*q3-q2*q4)],
               [2*(q2*q1-q3*q4),1-2*((q1**2)+(q3**2)),2*(q2*q3+q1*q4)],
               [2*(q3*q1+q2*q4),2*(q3*q2-q1*q4),1-2*((q1**2)+(q2**2))])
    return C

## DCM to Quaternions
def DCM2Q(C):
    q4=(1/2)*(1+C[0,0]+C[1,1]+C[2,2])
    C=numpy.array([[C[1,2]-C[2,1]],[C[2,0]-C[0,2]],[C[0,1]-C[1,0]]])
    q=(1/(1*4))*(C)
    return numpy.append(q,q4)    

## Quaternion error between two quaternion states q1 and q2
def q_error(qc,q1):
    # qc: commanded quaternion
    # q1: current quaternion
    q_e=numpy.zeros((4,1))
    q_temp=numpy.array([[qc[3],qc[2],-qc[1],-qc[0]],
                       [-qc[2],qc[3],qc[0],-qc[1]],
                       [qc[1],-qc[0],qc[3],-qc[2]],
                       [qc[0],qc[1],qc[2],qc[3]]])
    q_e_temp=numpy.matmul(q_temp,q1)
    q_e_n=math.sqrt((q_e_temp[0]**2)+(q_e_temp[1]**2)+(q_e_temp[2]**2)+(q_e_temp[3]**2))
    q_e=q_e_temp/q_e_n
    #print("norm 2",q_e_n,'\n')
    return q_e




# Kinematic differential equations

# Quaternion differential equation: related the angular velocity and quaternions.

# Using scipy.integrate.solve_ivp to integrate the differential equation
# function is written as dy/dt=f(t,y), y(t0)=y0

def Qdyn(t,q0,w):
    dy_dt=numpy.zeros((3,1))
    W=numpy.array([[0,w[2],-w[1],w[0]],[-w[2],0,w[0],w[2]],[w[2],-w[1],0,w[2]],[-w[0],-w[1],-w[2],0]])
    q0_n=numpy.linalg.norm(q0)
    q0=q0/q0_n
    dy_dt=(1/2)*numpy.matmul(W,q0)
    
    return dy_dt

# def W_Qdyn()