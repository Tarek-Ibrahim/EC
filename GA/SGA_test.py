import numpy as np
import sys, os
sys.path.append(os.path.abspath(".."))
# sys.path.append(os.path.abspath("../.."))

from SGA import SGA
from obj_funcs import *
from operators import *

# np.random.seed(5) #fix only for development
pop_size=10 #no. of individuals #rows
pc=0.75 #xover probability
pm=0.05 #mutation probability
T=500 #max no. of gens.
Ls=[5,5] #length of each variable in the chromosome
f=f3 #choice of the objective/test function
select=select_LR #parent selection operator
xover=xover_npt #crossover operator
fit=fit_bin
mutate=mutate_bflip

problem=SGA(pop_size,pc,pm,Ls,f)
problem.solve(fit,T,select,xover,mutate,decode)
