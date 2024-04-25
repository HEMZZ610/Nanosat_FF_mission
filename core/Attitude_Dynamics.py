"""

Attitude dynamics related functions

Date: 25-04-2024

Author:
    SDCS group
    Vishnuvardhan Shakthibala
"""
import numpy
import math

from Attitude_Kinematics import q_error

# General Euler equation of motion
# The dynamics is written in the pattern that is required for 
# integration using Scipy integration package

# Using scipy.integrate.solve_ivp to integrate the differential equation
# function is written as dy/dt=f(t,y), y(t0)=y0

## Attitude dynamics with generic moment of inertia
def attit_dyna_generic(t,w0,J,M):
    
    W=numpy.array([[0,-w0[2],w0[1]],
                   [-w0[2],0,-w0[0]],
                   [-w0[2],w0[1],0]])
    dy_dt=numpy.zeros([1,3])   
    dy_dt=numpy.matmul((1/J),(M-numpy.matmul(W,numpy.matmul(J,w0))))
    return numpy.reshape(dy_dt,(3,1))

## Attitude dynamics with principle moment of Intertia

def attit_dyna_principle(t,w0,J,M):
    dy_dt=numpy.zeros((3,))
    M=numpy.array([math.sin(t*0.24),math.cos(math.sin(t*0.064)),0])
    dy_dt[0]=((J[1,1]-J[2,2])/J[0,0])*w0[1]*w0[2]+M[0]
    dy_dt[1]=((J[2,2]-J[0,0])/J[1,1])*w0[2]*w0[0]+M[1]
    dy_dt[2]=((J[0,0]-J[1,1])/J[2,2])*w0[0]*w0[1]+M[2]
    
    return dy_dt

## Still need to test this
## N_flag is to indicate numerical (integrated) or analytical solution
## transverse_inertia_index indicates which index in the inertia matrix is transverse component.
def atti_dyna_axially_symmetric(t,w0,J,M,transverse_inertia_index,N_flag):
    dy_dt=numpy.zeros((3,))
    M=numpy.array([math.sin(t*0.24),math.cos(math.sin(t*0.064)),0])
    if (N_flag==1):
        if (transverse_inertia_index==1):
            dy_dt[0]=M[0]
            dy_dt[1]=((J[2,2]-J[0,0])/J[1,1])*w0[2]*w0[0]+M[1]
            dy_dt[2]=((J[0,0]-J[1.,1])/J[2,2])*w0[0]*w0[1]+M[2]
        elif (transverse_inertia_index==2):
            dy_dt[0]=((J[1,1]-J[2,2])/J[0,0])*w0[1]*w0[2]+M[0]
            dy_dt[1]=M[1]
            dy_dt[2]=((J[0,0]-J[1,1])/J[2,2])*w0[0]*w0[1]+M[2]
        else:
            dy_dt[0]=((J[1,1]-J[2,2])/J[0,0])*w0[1]*w0[2]+M[0]
            dy_dt[1]=((J[2,2]-J[0,0])/J[1,1])*w0[2]*w0[0]+M[1]
            dy_dt[2]=M[2]
        return dy_dt
    else:
        w=numpy.zeros((3,))
        if (transverse_inertia_index==1):
            wp=((J[0,0]/J[1,1])-1)*w0[0]
            w[0]=w0[0]
            w[1]=w0[1]*math.cos(wp*t)-w0[2]*math.sin(wp*t)
            w[2]=w0[2]*math.cos(wp*t)+w0[1]*math.sin(wp*t)
        elif (transverse_inertia_index==2):
            wp=((J[1,1]/J[2,2])-1)*w0[1]
            w[0]=w0[0]*math.cos(wp*t)-w0[2]*math.sin(wp*t)
            w[1]=w0[1]
            w[2]=w0[2]*math.cos(wp*t)+w0[0]*math.sin(wp*t)
        else:
            wp=((J[2,2]/J[0,0])-1)*w0[3]
            w[0]=w0[1]*math.cos(wp*t)-w0[0]*math.sin(wp*t)
            w[1]=w0[0]*math.cos(wp*t)+w0[1]*math.sin(wp*t)
            w[2]=w0[0]
        return w0


        
 


    
    
############# Miscelleneous ###################################

# Combine attitude dynamics and kinematics with principle moment of inertia

def attit_combined_principle(t,x,J,qc,K):
    
    dy_dt=numpy.zeros((7,))
    M=numpy.zeros((3,1))
    x[3:]=x[3:]/math.sqrt(x[3]**2+x[4]**2+x[5]**2+x[6]**2) # Normalizing 
    ## Quaternion error
    q_e=q_error(qc[3:],x[3:])

    ## H function
    H=1-q_e[3]
    term=numpy.matmul(J,x[0:3])
    ## control terms - Eigen axis control 
    M=numpy.cross(x[0:3],term)-K[0]*term-K[1]*numpy.matmul(J,q_e[0:3])
    
    
    ## Dynamics
    dy_dt[0]=((J[1,1]-J[2,2])/J[0,0])*x[1]*x[2]+(M[0]/J[0,0])
    dy_dt[1]=((J[2,2]-J[0,0])/J[1,1])*x[2]*x[0]+(M[1]/J[1,1])
    dy_dt[2]=((J[0,0]-J[1,1])/J[2,2])*x[0]*x[1]+(M[2]/J[2,2])
    
    ## kinematics represented in terms of quaternions
    W=numpy.array([[0,x[2],-x[1],x[0]],
                   [-x[2],0,x[0],x[1]]
                   ,[x[2],-x[1],0,x[2]],
                   [-x[0],-x[1],-x[2],0]])
    
    q0_n=math.sqrt((x[3]**2)+(x[4]**2)+(x[5]**2)+(x[6]**2))
    
    dy_dt[3:]=(1/2)*numpy.matmul(W,x[3:])

    return dy_dt
