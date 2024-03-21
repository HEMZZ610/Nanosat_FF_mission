"""
Usage:
    

Author:
    Vishnuvardhan Shakthibala
"""

## Copy the following lines of code 
# FROM HERE
import numpy
from scipy import integrate
import matplotlib.pyplot as plt
import os
import sys
import math
## ADD the packages here if you think it is needed and update it in this file.

## Import our libraries here
Attitude_library_path= os.path.join(os.path.dirname(os.path.abspath(__file__)),"../src")
sys.path.insert(0, Attitude_library_path)
import Attitude_Kinematics
import Attitude_Dynamics


## Copy Untill here



# Test case for M1:attitude_combined_principle present inside "Attitude_Dynamics" library

# Initial conditions
w0=numpy.array([0,0,0,0.233,-0.667,0.667,0.233]) # [angular velocity, quaternion]
J=numpy.array([[1500,0,0],[0,2700,0],[0,0,3000]]) # moment of intertia

## gains
k2=1.4
k1=math.sqrt(5*k2)

w0=numpy.array([0,0,0,0.233,-0.667,0.667,0.233])
w0[3:]=w0[3:]/math.sqrt(w0[3]**2+w0[4]**2+w0[5]**2+w0[6]**2)
J=numpy.array([[1500,0,0],[0,2700,0],[0,0,3000]])

final_cond=numpy.array([0,0,0,0,0,0,1]) # final desired states

t_span=[0,200]
K=numpy.array([k1,k2])

sol=integrate.solve_ivp(Attitude_Dynamics.attit_combined_principle, t_span, w0, method='RK45', args=(J,final_cond,K))


plt.figure(1)
plt.title('Quaternion vs time')
plt.plot(sol.t,sol.y[3] , 'k', linewidth=1, alpha=0.25)
plt.plot(sol.t,sol.y[4] , 'r', linewidth=1, alpha=0.25)
plt.plot(sol.t,sol.y[5] , 'y', linewidth=1, alpha=0.25)
plt.plot(sol.t,sol.y[6] , 'b', linewidth=1, alpha=0.25)

# plt.plot(np.zeros(w_events.shape[1]), w_events[1], 'r.', markersize=2.5, alpha=0.5)
plt.xlabel('time t')
plt.ylabel('quaternions')
plt.show()
