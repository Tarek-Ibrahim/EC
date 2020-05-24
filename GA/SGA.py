import numpy as np
from numpy import sin, cos, sqrt, pi, exp
from numpy.random import rand, randint
from matplotlib import pyplot as plt

class SGA:

    def __init__(self,sz,pc,pm,Ls,f):
        self.sz=sz
        self.pc=pc
        self.pm=pm
        self.Ls=Ls
        self.f=f
        self.L=np.sum(Ls) #total length of chromosome #columns
        self.best_fit=[]
        self.worst_fit=[]
        self.mean_fit=[]

    def vis(self,pop,fitness,opt_fit,opt_obj,t,decode,best_off,fit_prev,offspring):
        
        np.set_printoptions(suppress=True)

        if np.any(fitness==opt_fit):
            opt_sol=decode(offspring,self.Ls)[np.where(fitness==opt_fit)[0][0],:]
            est_fit=opt_fit
        else:
            opt_sol=best_off
            est_fit=fit_prev

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

    def solve(self,fit,T,select,xover,mutate,decode):

        t=0
        pop=randint(2, size=(self.sz,self.L)); offspring=pop; #population is constructed/Initialized and used as a matrix
        fitness,opt_obj=fit(pop,self.Ls,self.f); opt_fit=1/(opt_obj[-1]+1); #adding 1 to denominator to avoid division by zero
        self.best_fit.append(np.max(fitness)); self.worst_fit.append(np.min(fitness)); self.mean_fit.append(np.mean(fitness));
        best_off=float('nan'); fit_prev=-np.inf

        while not(np.any(fitness==opt_fit) or t==T):
            parents=select(offspring,fitness)
            offspring=xover(parents,self.pc)
            offspring=mutate(offspring,self.pm)
            fitness,_=fit(offspring,self.Ls,self.f)
            
            if np.max(fitness)>fit_prev:
                best_off=decode(offspring,self.Ls)[np.argmax(fitness),:] 
                fit_prev=np.max(fitness)
            self.best_fit.append(np.max(fitness)); self.worst_fit.append(np.min(fitness)); self.mean_fit.append(np.mean(fitness))
            t+=1
        
        self.vis(pop,fitness,opt_fit,opt_obj,t,decode,best_off,fit_prev,offspring)