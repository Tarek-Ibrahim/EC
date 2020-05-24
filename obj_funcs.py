import numpy as np
from numpy import array, sin, cos, sqrt, pi, exp


def F1(var):
    #-pi<=x1, x2 <=pi
    ival=[-pi,pi]
    x1=var[0]; x2=var[1]
    
    A1=0.5*sin(1)-2*cos(1)+sin(2)-1.5*cos(2)
    A2=1.5*sin(1)-cos(1)+2*sin(2)-0.5*cos(2)
    B1=0.5*sin(x1)-2*cos(x1)+sin(x2)-1.5*cos(x2)
    B2=1.5*sin(x1)-cos(x1)+2*sin(x2)-0.5*cos(x2)
    
    f1=1+(A1-B1)**2+(A2-B2)**2
    f2=(x1+3)**2+(x2+1)**2
    fv=array([f1,f2])
    
    return fv, ival

def F2(var):
    #-3<=x1, x2 <=3
    ival=[-3,3]
    x1=var[0]; x2=var[1]
    
    f1=0.5*(x1**2+x2**2)+sin(x1**2+x2**2)
    f2=(3*x1-2*x2+4)**2/8+(x1-x2+1)**2/27+15
    f3=1/(x1**2+x2**2+1)-1.1*exp(-(x1**2+x2**2))
    fv=array([f1,f2,f3])
    
    return fv, ival

def fa(var):#ackley (nD)
    #f(0)=0
    #-30<=x_j<=30 (var domain)
    obj=-20*exp(-0.2*sqrt((1/len(var))*np.sum(var**2)))-exp((1/len(var))*np.sum(cos(2*pi*var)))+20+exp(1)
    opt=[0]*len(var)+[0]
    return obj, opt

def fg(var): #griewangk (nD)
    #f(0)=0
    #-600<=x_j<=600 (var domain)
    obj=(1/4000)*np.sum(var**2)-np.prod(cos(var/sqrt(np.ones(len(var))*range(len(var))+1))+1)
    opt=[0]*len(var)+[0]
    return obj, opt
    
def f1(var): #1D
    #f(0)=1
    obj=np.abs(var)+cos(var) #evaluation of objective function
    opt=[0,1] #(value of function's variables at the optimum, known optimal objective function value)
    return obj, opt

def f3(var): #nD
    #f(0)=0
    obj=np.sum(var**2)
    opt=[0]*len(var)+[0]
    return obj, opt

def f10(var): #nD
    #f(0)=0
    obj=10*len(var)+np.sum(var**2-10*cos(2*pi*var))
    opt=[0]*len(var)+[0]
    return obj, opt

def f12(var): #2D
    #f(1.897,1.006) = -0.5231
    #f(0)=0 #corrected
    obj=0.5+((sin(sqrt(var[0]**2+var[1]**2))**2-0.5)/(1+0.1*(var[0]**2+var[1]**2)))
    opt=[0]*len(var)+[0]
#     opt=[1.897,1.006,-0.5231]
    return obj, opt






