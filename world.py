import numpy as np
from numpy.random import rand, randint
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from skimage.draw import line

def can_connect(a,b,grid,occ):
        connection = line(a[0], a[1], b[0], b[1])  
        if np.any(grid[connection] == occ):
            return False

        return True

def init_grid(nodes,eg,occ,n,ip,max_conn,can_connect,am=True):
    graph=np.eye(n)*ip
    graph[graph == 0] = np.inf

    for i in range(n):
        for j in range(n):
            if (j!=i and can_connect(nodes[i,:],nodes[j,:],eg,occ)):
                a = np.atleast_2d(nodes[i,:])
                b = np.atleast_2d(nodes[j,:])
                graph[i,j]=np.linalg.norm(a - b, axis=1) if am else ip


        idx = np.argpartition(graph[i,:], max_conn)
        graph[i,idx[max_conn:]] = np.inf

    graph[graph==np.inf]=0
    return graph

def plot_grid(eg,sz,idx,nodes,grid,path=[],name=None):
    plt.figure(figsize=(6,6))

    #Grid
    plt.imshow(eg.T,cmap=plt.cm.Greys)
    ax=plt.gca()
    ax.grid(color='k', linestyle='-', linewidth=2)
    ax.set_xticks(np.arange(-0.5, 10, 1))
    ax.set_yticks(np.arange(-0.5, 10, 1))
    ax.set_xticklabels(np.arange(0,10, 1))
    ax.set_yticklabels(np.arange(0, 10, 1))

    #Nodes
    plt.scatter(idx[0],idx[1])
    for node, coord in enumerate(nodes):
        plt.text(coord[0], coord[1], str(node))
        
    #Connections
    source, sink = np.nonzero(grid)
    source=nodes[source,:]
    sink=nodes[sink,:]
    lc = LineCollection(np.stack((source, sink), axis=1),linewidths=[1], colors=[(0, 0.75, 1, 1)])
    ax.add_collection(lc)

    #Paths
    if path!=[]:
        ax.plot(path[:, 0], path[:, 1], 'ro-', linewidth=2,label=name)

    plt.legend()
    plt.show()