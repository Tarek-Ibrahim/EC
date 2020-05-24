import numpy as np
import sys, os
sys.path.append(os.path.abspath(".."))
# sys.path.append(os.path.abspath("../.."))

from PPA import PPA
from obj_funcs import *
from operators import *


# np.random.seed(20)

sz=30 #[20,20] #100 #lattice size
pc=1 #crossover parameters
pm=0.2 #F=0.7; #mutation parameters
pv=0.5 #move probability
f=F1 #choice of group of objective functions
n=2 #number of variables
T=200 #3000 #number of evaluations
preys=240 #initial prey population size
preds=20 #number of predators
preys_pref=120 #preferred number of preys 
mutate=mutate_replace
xover1=xover_1pt
xover2=xover_blx

problem=PPA(sz,pv,pm,pc,preds,preys_pref,f,xover1,mutate)
problem.solve(T,n,preys)