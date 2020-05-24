import numpy as np
import random
from numpy import sin, cos, sqrt, pi, exp
from numpy.random import rand, randint, uniform
from matplotlib import pyplot as plt

class DE:

    def __init__(self,sz,b,f,pc,F):
        self.sz=sz
        self.b=b
        self.f=f
        self.pc=pc
        self.F=F
        self.best_fit=[]
        self.worst_fit=[]
        self.mean_fit=[]
    
    def vis(self,pop,fitness,opt_fit,opt_obj,t):
        np.set_printoptions(suppress=True)

        if np.any(fitness==opt_fit):
            opt_sol=pop[np.where(fitness==opt_fit)[0][0],:]
            est_fit=opt_fit
        else:
            opt_sol=pop[np.argmax(fitness),:]
            est_fit=np.max(fitness)

        best_obj,_=self.f(opt_sol)

        print("total number of generations elapsed = ",t, '\n')

        # print("final fitness vector = ", fitness,'\n')
        # print("optimal fitness value = ", opt_fit,'\n')
        # print("best found fitness value = ",est_fit ,'\n')

        print("optimal objective function value = ", opt_obj[-1],'\n')
        print("best found objective function value = ",best_obj ,'\n')

        print("variables values at true optimum =",opt_obj[0:-1] ,'\n')
        print("variables values at found optimum =",opt_sol ,'\n')

        plt.figure(figsize=(8,6))
        plt.plot(self.best_fit,label='best fitness')
        plt.plot(self.mean_fit,label='mean fitness')
        plt.plot(self.worst_fit,label='worst fitness')
        plt.legend()
        plt.xlabel("number of generations")
        plt.ylabel("fitness value")
        plt.show()
    
    def solve(self,T,n,fit,xover,mutate,select):

        t=0
        pop=uniform(self.b[0],self.b[1],(self.sz,n))
        fitness,opt_obj=fit(pop,self.f); opt_fit=1/(opt_obj[-1]+1)
        self.best_fit.append(np.max(fitness)); self.worst_fit.append(np.min(fitness)); self.mean_fit.append(np.mean(fitness))
        
        while not(np.any(fitness==opt_fit) or t==T):
            pop_mu=mutate(pop,self.F) #mutation matrix
            pop_T=xover(pop,pop_mu,self.pc) #trial matrix
            pop=select(pop,pop_T,self.f,fit)
            fitness,_=fit(pop,self.f)
            self.best_fit.append(np.max(fitness)); self.worst_fit.append(np.min(fitness)); self.mean_fit.append(np.mean(fitness));
            t+=1
        
        self.vis(pop,fitness,opt_fit,opt_obj,t)