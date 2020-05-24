import numpy as np
from numpy.random import rand, randint

class ACO:

    def __init__(self,grid,start,end,m,alpha,beta,rho,Q):
        self.grid=grid
        self.start=start
        self.end=end
        self.m=m
        self.alpha=alpha
        self.beta=beta
        self.rho=rho
        self.Q=Q

    def select_aco(self,pop,prob,r): #RW
        cum_prob=np.cumsum(prob)
        idx=np.where(cum_prob>rand())[0][0]
        return r[idx]

    def ant_sol(self,nodes,n):
        s=[]; L=np.zeros(self.m)
        for k in range(self.m):
            i=self.start; sol=[]
            while(i!=self.end):
                a=nodes[i,:]
                r=np.nonzero(self.grid[i,:])[0]
                r=np.asarray([e for e in r if e!=i])
                b=nodes[r,:]
                eta=1/np.linalg.norm(a - b, axis=1); tao=self.grid[i,r]
                p=((tao**self.alpha)*(eta**self.beta))/sum((tao**self.alpha)*(eta**self.beta))
                p=p[np.nonzero(p)[0]]
                idx=self.select_aco(nodes[r,:],p,r)
                sol.append((i,idx))
                i=idx
            L[k]=len(sol)
            s.append(sol)
        
        return s,L

    def update_phermones(self,s,L,n):
        for i in range(n):
            for j in range(n):
                if (i!=j and self.grid[i,j]!=0):
                    delta=[self.Q/L[k] if ([item for item in s[k] if (item[0]==i and item[1]==j)]!=[]) else 0 for k in range(self.m)]
                    self.grid[i,j]=(1-self.rho)*self.grid[i,j]+sum(delta)

    def get_path(self,s,L):
        path_aco=s[np.argmin(L)]
        path_aco=[i[0] for i in path_aco]+[path_aco[-1][-1]]
        
        cost=[self.grid[path_aco[i],path_aco[i+1]] for i in range(len(path_aco)-1)]  
        cost_aco=sum(cost)
        
        return path_aco,cost_aco

    def solve(self,T,nodes,n):
        for t in range(T):
            s,L=self.ant_sol(nodes,n)
            self.update_phermones(s,L,n)

        path_aco,cost_aco=self.get_path(s,L)
        path=nodes[path_aco]

        print("ACO Path = ", path_aco)
        print("ACO Path Total Cost = ", cost_aco)

        return path