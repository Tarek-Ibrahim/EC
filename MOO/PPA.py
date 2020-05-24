import numpy as np
import random
from numpy.random import rand, randint, uniform
from matplotlib import pyplot as plt

# TODO:
# 1- non-square lattice
# 2- combined 1pt and blx xover
# 3- penalizing infeasible children (constraint handling)
# 4- 3D pareto plot (and handling larger dimensions in plot)
# 5- [Optional] Animation of lattice evolution

class pop:
    
    def  __init__(self,name,value):
        self.name=name
        self.value=value

        
class lattice:
    
    def  __init__(self,sz,pv,pm,pc,ival,preds,preys_pref,f,xover,mutate):
        self.grid=np.zeros((sz,sz)).astype(int).tolist()
        self.sz=sz
        self.pv=pv
        self.pm=pm
        self.pc=pc
        self.ival=ival
        self.preds=preds
        self.pf=preys_pref
        self.f=f
        self.xover=xover
        self.mutate=mutate
        
    def distrib(self,pop):
        for i in pop:
            r=randint(0,self.sz,2)
            while(self.grid[r[0]][r[1]]!=0):
                r=randint(0,self.sz,2)
            self.grid[r[0]][r[1]]=i
            
    def vis(self,title):
        
        idx=np.nonzero(self.grid)
        sz=self.sz
        preys=[[idx[0][i],idx[1][i]] for i in range(len(idx[0])) if self.grid[idx[0][i]][idx[1][i]]!=0 and self.grid[idx[0][i]][idx[1][i]].name=="prey"]
        preds=[[idx[0][i],idx[1][i]] for i in range(len(idx[0])) if self.grid[idx[0][i]][idx[1][i]]!=0 and self.grid[idx[0][i]][idx[1][i]].name=="pred"]
        preys=np.asarray(preys)
        preds=np.asarray(preds)

        plt.figure(figsize=(7,7))
        plt.imshow(np.zeros((sz,sz)),cmap=plt.cm.Greys)
        ax=plt.gca()
        ax.grid(color='k', linestyle='-', linewidth=2)
        ax.set_xticks(np.arange(-0.5, sz, 1))
        ax.set_yticks(np.arange(-0.5, sz, 1))
        ax.set_xticklabels(np.arange(0,sz, 1))
        ax.set_yticklabels(np.arange(0, sz, 1))

        plt.scatter(preys[:,0],preys[:,1],color='b',label="preys")
        plt.scatter(preds[:,0],preds[:,1],color='r',label="preds")
        
        plt.title(title)
        plt.legend()
        plt.show()
        
    def move(self,name):
        idx=np.nonzero(self.grid)
        sz=self.sz
        n=len(idx[0])
        prob=rand(n)
        for i in range(n):
            cell=self.grid[idx[0][i]][idx[1][i]]
            if cell.name==name and prob[i]<=self.pv:
                count=0; m=[0,0]; j=0; k=0
                while((self.grid[j][k]!=0 or (m[0]==0 and m[1]==0)) and count!=10):
                    m=randint(-1,2,2)
                    j=sz-1 if (idx[0][i]==0 and m[0]==-1) else 0 if (idx[0][i]==sz-1 and m[0]==1) else idx[0][i]+m[0]
                    k=sz-1 if (idx[1][i]==0 and m[1]==-1) else 0 if (idx[1][i]==sz-1 and m[1]==1) else idx[1][i]+m[1]
                    count+=1
                if count==10 and self.grid[j][k]!=0:
                    continue
                self.grid[j][k]=cell
                self.grid[idx[0][i]][idx[1][i]]=0
        
    def breed(self):
        idx=np.nonzero(self.grid)
        sz=self.sz
        n=len(idx[0])
        m=[-1,0,1]
        for i in range(n):
            p1=self.grid[idx[0][i]][idx[1][i]]
            if p1.name=="prey":
                nbhd=[]
                for r in m:
                    for c in m:
                        if not(r==0 and c==0):
                            j=sz-1 if idx[0][i]==0 and r==-1 else 0 if idx[0][i]==sz-1 and r==1 else idx[0][i]+r
                            k=sz-1 if idx[1][i]==0 and c==-1 else 0 if idx[1][i]==sz-1 and c==1 else idx[1][i]+c
                            nbhd.append(self.grid[j][k])
                nbhd=[nb for nb in nbhd if nb!=0 and nb.name=="prey"]
                if nbhd:
                    p2=random.sample(nbhd,1)[0]
                    ch=self.xover(np.array([p1.value,p2.value]),self.pc) #need the 2nd xover operator to produce the 2nd half of the pop
                    ch=self.mutate(ch,self.pm,self.ival)[0,:]
                    ch=pop("prey",ch)
                    rn=randint(0,sz,2); count=0
                    while(self.grid[rn[0]][rn[1]]!=0 and count!=10):
                        rn=randint(0,sz,2)
                        count+=1
                    if count==10 and self.grid[rn[0]][rn[1]]!=0:
                        continue
                    self.grid[rn[0]][rn[1]]=ch
        
    def kill(self):
        idx=np.nonzero(self.grid)
        sz=self.sz
        n=len(idx[0])
        pn=sum([1 for i in range(n) if self.grid[idx[0][i]][idx[1][i]].name=="prey"])

        T=int(np.floor((pn-self.pf)/self.preds))
        for t in range(T):
            idx=np.nonzero(self.grid)
            n=len(idx[0])
            m=[-1,0,1]
            for i in range(n):
                cell=self.grid[idx[0][i]][idx[1][i]]
                if cell!=0 and cell.name=="pred":
                    nbhd=[]; idcs=[]
                    for r in m:
                        for c in m:
                            if not(r==0 and c==0):
                                j=sz-1 if idx[0][i]==0 and r==-1 else 0 if idx[0][i]==sz-1 and r==1 else idx[0][i]+r
                                k=sz-1 if idx[1][i]==0 and c==-1 else 0 if idx[1][i]==sz-1 and c==1 else idx[1][i]+c
                                if self.grid[j][k]!=0 and self.grid[j][k].name=="prey":
                                    nbhd.append(self.grid[j][k])
                                    idcs.append([j,k])
                    if nbhd:
                        fitness=[np.sum(self.f(i.value)[0]*cell.value) for i in nbhd]
                        lf=np.argmin(fitness)
                        idx2=idcs[lf]
                        self.grid[idx2[0]][idx2[1]]=cell
                        self.grid[idx[0][i]][idx[1][i]]=0
                    else:
                        self.move("pred")
    
    def pareto(self):
        
        idx=np.nonzero(self.grid)
        pareto_v=np.asarray([self.grid[idx[0][i]][idx[1][i]].value for i in range(len(idx[0])) if self.grid[idx[0][i]][idx[1][i]]!=0 and self.grid[idx[0][i]][idx[1][i]].name=="prey"])
        pareto_f=np.asarray([self.f(i)[0] for i in pareto_v])
        
        plt.figure(figsize=(7,7))
        plt.scatter(pareto_f[:,0],pareto_f[:,1],color='b')
        
        plt.ylabel("f1")
        plt.xlabel("f2")
        plt.title("Pareto Front")
        plt.show()
        
        return pareto_v, pareto_f

class PPA:
    
    def  __init__(self,sz,pv,pm,pc,preds,preys_pref,f,xover,mutate):
        self.sz=sz
        self.pv=pv
        self.pm=pm
        self.pc=pc
        self.preds=preds
        self.pf=preys_pref
        self.f=f
        self.xover=xover
        self.mutate=mutate

    def solve(self,T,n,preys):
        fv,ival=self.f(np.asarray([0]*n))
        grid=lattice(self.sz,self.pv,self.pm,self.pc,ival,self.preds,self.pf,self.f,self.xover,self.mutate)

        prey=uniform(ival[0],ival[1],(preys,n))
        prey=[pop("prey",prey[i]) for i in range(preys)]
        grid.distrib(prey)

        pred=np.asarray([random.sample(list(np.arange(0,1+1/self.preds,1/(self.preds-1))), self.preds) for i in range(len(fv))]).T
        pred=[pop("pred",pred[i]) for i in range(self.preds)]
        grid.distrib(pred)

        grid.vis("Initial Configuration")

        for t in range(T):
            grid.move("prey")
            grid.breed()
            grid.kill()

        grid.vis("Final Configuration")
        pareto_v, pareto_f=grid.pareto()

        return pareto_v, pareto_f
