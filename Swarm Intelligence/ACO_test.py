import numpy as np
import sys, os
sys.path.append(os.path.abspath(".."))
# sys.path.append(os.path.abspath("../.."))

from ACO import ACO
from obj_funcs import *
from operators import *
from world import *

np.random.seed(20)

#Grid Parameters
occ=100
max_conn=10
sz=10
ip=0.1

#Initialize Empty Grid
eg=np.zeros((sz,sz))

#Defining Obstacles
eg[1,7:10]=occ; eg[2,7:9]=occ; eg[3,1:4]=occ;
eg[3:8,5]=occ; eg[4,1]=occ; eg[7,1:4]=occ;
eg[6:9,8]=occ; eg[6,7]=occ;

#Defining Nodes (Discretization)
idx=np.where(eg!=occ)
n=idx[0].shape[0]
nodes=[[idx[0][i],idx[1][i]] for i in range(len(idx[0]))]
nodes=np.asarray(nodes)

#Initialize Grid
grid=init_grid(nodes,eg,occ,n,ip,max_conn,can_connect)

#Path
start=8
end=60

#ACO
rho=0.8
alpha=1
beta=1
Q=1
m=50
T=20

problem=ACO(grid,start,end,m,alpha,beta,rho,Q)
path=problem.solve(T,nodes,n)

plot_grid(eg,sz,idx,nodes,grid,path,"ACO")