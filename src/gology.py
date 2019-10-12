#!/usr/bin/env python
# coding: utf-8

# In[458]:


import numpy as np
import concurrent.futures as ccf
from numba import njit

@njit
def symmetry(i,N):
    i4 = (i[0]+i[1]-i[2])%N
    mi4 = (-i4)%N
    mi1 = (-i[0])%N
    mi2 = (-i[1])%N
    mi3 = (-i[2])%N
    all_ = []
    test = {(mi1,mi2,mi3),
            (mi2,mi1,mi4),
            (mi3,mi4,mi1),
            (i[1],i[0],i4),
            (i[2],i4, i[0]),
            (i4,i[2],i[1]),
            (mi4,mi3,mi2)
               }
    
    for m in test:
        if m != i:
            all_.append(m)
    return all_

@njit
def getInd(N):
    indices =[
            (i,j,k)
            for i in range(N)
            for j in range(N)
            for k in range(N)
        ]
    for ind in indices:
        for x in symmetry(ind,N=N):
            if x in indices:
                indices.remove(x)  
    return indices

class Gology:
    def __init__(self,N,g1=0,g2=0,g3=0):
        self.N = N
        self.g1 = np.ones((N,N,N), float)*g1
        self.g2 = np.ones((N,N,N), float)*g2
        self.g3 = np.ones((N,N,N), float)*g3
        self.get_indices()
    
    def get_indices(self):
        self.indices= tuple(getInd(self.N))
    
    def get_indices_Old(self):
        self.indices =[
            (i,j,k)
            for i in range(self.N)
            for j in range(self.N)
            for k in range(self.N)
        ]
        for ind in self.indices:
            [self.indices.remove(x) for x in self.symmetry(ind) if x in self.indices]
        self.indices = tuple(self.indices)

    def __getitem__(self,i):
        return (self.g1[i],self.g2[i],self.g3[i])

    def symmetry(self,*j):
        i=j[0]
        i4 = (i[0]+i[1]-i[2])%self.N
        mi4 = (-i4)%self.N
        mi1 = (-i[0])%self.N
        mi2 = (-i[1])%self.N
        mi3 = (-i[2])%self.N
        all_ = ()
        test = {(mi1,mi2,mi3),
            (mi2,mi1,mi4),
            (mi3,mi4,mi1),
            (i[1],i[0],i4),
            (i[2],i4, i[0]),
            (i4,i[2],i[1]),
            (mi4,mi3,mi2)
               }
        all_ = tuple(m for m in test if m != i)
        return all_
    
    def __contains__(self, item):
        return item in self.indices
        
    @property
    def get(self):
        a = np.zeros((self.N,self.N,self.N), dtype=bool)
        for p in self.indices:
            a[p]=True
        return a
            
    
    def pack(self):
        y = np.concatenate(
            (self.g1[self.get],
             self.g2[self.get],
             self.g1[self.get]
            )
        )
        return y 
    
    def unpack(self,y):
        n = len(self.indices)
        self.g1[self.get] = y[:n]
        self.g2[self.get] = y[n:2*n]
        self.g3[self.get] = y[2*n:]
        
        for k in self.indices:
            for j in self.symmetry(k):
                self.g1[j]= self.g1[k]
                self.g2[j]= self.g2[k]
                self.g3[j]= self.g3[k]
    
    def __iter__(self):
        return iter(self.indices)
        
        


# In[459]:


a =Gology(32)


# In[461]:


len(a.indices)


# In[ ]:




