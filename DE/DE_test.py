import numpy as np
import sys, os
sys.path.append(os.path.abspath(".."))

from DE import DE
from obj_funcs import *
from operators import *

# np.random.seed(5) #fix only for development
n=2 #number of variables
pop_size=10*n*2 #no. of individuals >=10*n
pc=0.3 #xover probability
F=0.7 #mutation scale factor
T=500 #max no. of gens.
t=0 #gen. number
b=[-30,31] #initial pop bounds
f=f12 #choice of the objective/test function
fit=fit_real
xover=xover_de
mutate=mutate_diff
select=select_de

problem=DE(pop_size,b,f,pc,F)
problem.solve(T,n,fit,xover,mutate,select)