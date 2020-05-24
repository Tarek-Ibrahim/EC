import numpy as np
import random
from numpy.random import rand, randint, uniform

def mutate_diff(pop,F): #differential mutation
    r=np.asarray([random.sample(list(range(0,i))+list(range(i+1,pop.shape[0])),3) for i in range(pop.shape[0])])
    pop_mu=np.asarray([pop[r[i,0],:]+F*(pop[r[i,1],:]-pop[r[i,2],:]) for i in range(pop.shape[0])])
    return pop_mu

def xover_de(pop,pop_mu,pc):
    k=randint(pop.shape[1]+1,size=pop.shape[0])
    U=rand(pop.shape[0],pop.shape[1])
    pop_T=np.asarray([[pop_mu[i,j] if (U[i,j]<pc or j==k[i]) else pop[i,j] for j in range(pop.shape[1])] for i in range(pop.shape[0])])
    return pop_T

def select_de(pop,pop_T,f,fit):
    fit_pop,_=fit(pop,f); fit_pop_T,_=fit(pop_T,f)
    pop_new=np.asarray([pop[i,:] if fit_pop[i]>fit_pop_T[i] else pop_T[i,:] for i in range(pop.shape[0])])
    return pop_new

def fit_real(pop,f):
    pop_obj=np.asarray([f(pop[i,:])[0] for i in range(pop.shape[0])])
    pop_fit=1/(pop_obj+1)
    _,opt_obj=f(pop[0,:])
    return pop_fit, opt_obj

def gamma(dec,x,y,Ls,dim): 
    """ maps decimal representation of an input binary string to a real value in the range [x,y] """
    return x+((y-x)/(2**Ls[dim]-1))*dec

def decode(a,Ls):
    """decodes binary representation to integer/real number"""
    sz=np.size(a,0) #no. of individuals
    n=len(Ls) #no. of dimensions
    mat=np.zeros((sz,n))
    L=np.cumsum(Ls)
    for dim in range(n):
        a_trim=a[:,:L[dim]] if dim==0 else a[:,L[dim-1]:L[dim]]
        col=a_trim.dot(1 << np.arange(a_trim.shape[-1] - 1, -1, -1)) #converting each variable/dimension seperately
#         col=gamma(col,-1,62,Ls,dim) #uncomment if mapping to real values is desired
        mat[:,dim]=col
    return mat

def fit_bin(pop,Ls,f): # TODO: handle -ve values for f(x)
    """returns pop fitnesses (i.e. fitness vector)"""
    n=np.size(pop,0) #no. of individuals
    mat=decode(pop,Ls)
    pop_obj=np.zeros(n) #objective function values for population individuals
    for ir in range(n):
        obj,_=f(mat[ir,:])
        pop_obj[ir]=obj
    pop_fit=1/(pop_obj+1)   #add 1 to denom. to avoid divsion by zero
    _,opt_obj=f(mat[0,:])
    
    return pop_fit, opt_obj

def select_RW(pop,fitness,scaled=False): #SUS?
    """RW selection with an option to sigma scale the fitness"""
    fitness=sigma_scale(fitness) if scaled else fitness
    prob=fitness/np.sum(fitness)
    cum_prob=np.cumsum(prob)
    idx=[np.where(cum_prob>rand())[0][0] for i in range(np.size(pop,0))]
    return pop[idx,:]

def sigma_scale(fitness): #sigma scaling
    c=2.; mf=np.mean(fitness); sigma=np.std(fitness)
    fitness_scaled=np.zeros(len(fitness))
    for i,f in enumerate(fitness):
        fitness_scaled[i]=np.max([f-(mf-c*sigma),0])
    return fitness_scaled

def select_LR(pop,fitness,s=1.5):
    """linear ranking-based selection"""
    mu=np.size(pop,0)
    idx=np.argsort(fitness)
    pops=pop[idx,:] #sorted pop according to fitness
    prob=[((2-s)/mu)+((2*i*(s-1))/(mu*(mu-1))) for i in range(mu)]
    cum_prob=np.cumsum(prob)
    idx=[np.where(cum_prob>rand())[0][0] for i in range(mu)]
    return pops[idx,:]

def select_tour(pop,fitness,k=2):
    """tournament selection with replacement"""
    champ=np.zeros_like(pop) #tournament champions
    for i in range(np.size(pop,0)):
        idx=randint(low=0,high=np.size(pop,0),size=k)
        best_f=np.max(fitness[idx])
        champ[i,:]=pop[np.where(fitness==best_f)[0][0],:]
    return champ

def select_elite(): #2 if even 1 if odd 
    #parents that get copied to next gen. w/o xover or mutation (survivor selection method)
    #TODO
    pass

def mutate_replace(pop,pm,ival):
    """ replaces a gene in a chromosome with another floating point 
    number randomly chosen within the bounds of the parameter values"""
    prob=rand(pop.shape[0])
    gene=[randint(pop.shape[1]) for i in prob if i<pm] #choosing genes (gene indicies) to replace
    gene_mu=uniform(ival[0],ival[1],len(gene)) #the replacements
    for i in range(len(gene)):
        pop[np.where(prob<pm)[0][i],gene[i]]=gene_mu[i]
    return pop

def xover_npt(pop,pc,n=2):
    sz=pop.shape; m=int(np.round(sz[0]/2)); L=sz[1];
    prob=rand(m)
    cop=[0]*n
    for i in range(m):
        if prob[i]<pc:
            while not(len(cop)==len(set(cop))):
                cop=(np.round(rand(n)*(L-1))+1).astype(int)
                cop=cop.tolist()
            cop=np.sort(np.asarray(cop))
            p1=pop[2*i,:] #parent 1
            p2=pop[0,:] if i==n-1 else pop[2*i+1,:] #parent 2
            for j in range(len(cop)):
                if (len(cop)%2!=0 and j==len(cop)-1):
                    t=p1[cop[j]:]
                    p1[cop[j]:]=p2[cop[j]:]
                    p2[cop[j]:]=t
                elif ((len(cop)%2==0 and j%2==0) or (len(cop)%2!=0 and j!=len(cop)-1)):
                    t=p1[cop[j]:cop[j+1]]
                    p1[cop[j]:cop[j+1]]=p2[cop[j]:cop[j+1]]
                    p2[cop[j]:cop[j+1]]=t
            pop[2*i,:]=p1
            if i==m-1:
                pop[0,:]=p2
            else:
                pop[2*i+1,:]=p2
            
    return pop

def mutate_bflip(pop,pm):
    """bit-flipping mutation"""
    sz=pop.shape
    prob=rand(sz[0],sz[0])
    idx=np.unravel_index(np.where(prob<pm)[0],sz)
    r=idx[0]; c=idx[1];
    for i in range(len(r)):
        pop[r[i],c[i]]=~pop[r[i],c[i]]+2
    
    return pop

def xover_1pt(pop,pc):
    """1 point xover by swapping tails after crossover point"""
    sz=pop.shape; n=int(np.round(sz[0]/2)); L=sz[1]
    prob=rand(n)
    cop=(np.round(rand(n)*(L-1))+1).astype(int) #random crossover points generation
    for i in range(n):
        if prob[i]<pc:
            p1=pop[2*i,:] #parent 1
            p2=pop[0,:] if i==n-1 else pop[2*i+1,:] #parent 2
            t=p1[cop[i]:]
            p1[cop[i]:]=p2[cop[i]:]
            p2[cop[i]:]=t
            pop[2*i,:]=p1
            if i==n-1:
                pop[0,:]=p2
            else:
                pop[2*i+1,:]=p2
    return pop

def xover_uniform_real(pop,pc):
    sz=pop.shape; n=int(np.round(sz[0]/2)); L=sz[1]
    prob=rand(n)
    for i in range(n):
        if prob[i]<pc:            
            p1=pop[2*i,:] #parent 1
            p2=pop[0,:] if i==n-1 else pop[2*i+1,:] #parent 2
            c1=np.zeros(len(p1))
            c2=np.zeros(len(p2))
            coin=(np.round(rand(L))).astype(int)
            for j in range(L):
                c1[j]=p1[j] if coin[j]==1 else p2[j]
                c2[j]=p2[j] if coin[j]==1 else p1[j]
            pop[2*i,:]=c1
            pop[2*i+1,:]=c2
            
    return pop

def xover_uniform_bin(pop,pc):
    sz=pop.shape; n=int(np.round(sz[0]/2)); L=sz[1]
    prob=rand(n)
    for i in range(n):
        if prob[i]<pc:            
            p1=pop[2*i,:] #parent 1
            p2=pop[0,:] if i==n-1 else pop[2*i+1,:] #parent 2
            c1=np.zeros(len(p1))
            c2=np.zeros(len(p2))
            coin=(np.round(rand(L))).astype(int)
            for j in range(L):
                c1[j]=p1[j] if coin[j]==1 else p2[j]
            c1=c1.astype(int)
            c2=~c1+2
            pop[2*i,:]=c1
            pop[2*i+1,:]=c2
            
    return pop

def xover_blx(pop,alpha=0.5):
    """BLX-alpha crossover"""
    n=int(np.round(pop.shape[0]/2))
#     gamma=uniform(-alpha,alpha+1,n)
    mu=rand(n); gamma=(1+2*alpha)*mu-alpha
    pop=np.asarray([gamma*pop[2*i,:]+(1-gamma)*pop[2*i+1,:] for i in range(n)])
    return pop

