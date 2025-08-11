# -*- coding: utf-8 -*-
"""
Created on Mon Aug 11 00:50:40 2025

@author: Michael
"""
from opti import Opt
import numpy as np
import scipy as sp

M=10000
class Factory:
    def __init__(self):
        self.plants=[]
        
    def addPlant(self, plant, amount):
        self.plants.append((plant,amount))
        
class Product:
    def __init__(self,name,factory, assignments):
        self.factory=factory
        self.assignments=assignments
        self.name=name
        
        
class Scheduling:
    def __init__(self, factory):
        self.factory=factory
        self.Schedule=[]
        
    def addProduct(self, product):
        if self.factory==product.factory:
            self.Schedule.append(product)
    
    def createProblem(self):
        p=Opt()
        k=1
        for i in range(len(self.Schedule)):
            for j in range(len(self.factory.plants)):
                x=f'x_{i,j}'
                s=f's_{i,j}'
                s_old=f's_{i,j-1}'
                size=self.factory.plants[j][1]
                p.addVar(x, size) 
                p.addVar(s,1)    
                
                p.addIntegrality(x, 1)
                p.setBound(x, 0, 1)
                p.setBound(s, 0, np.inf)
                
                
                one=f'one_{i,j}'       
                p.addLinConstr(one, 1)
                p.setSubMatrix(x, one, np.ones((1,size)))
                p.setLinBound(one, 1, 1)
                
                if j>0:
                    selfInt=f'self_{i}'
                    p.addLinConstr(selfInt, 1)
                    p.setSubMatrix(s, selfInt, -1)
                    
                    p.setSubMatrix(s_old, selfInt, 1)
                    
                    p.setLinBound(selfInt, -np.inf, -self.Schedule[i].assignments[j-1])
                    
                
            for l in range(i):
                for j in range(len(self.factory.plants)):
                    overlap=f'Overlap_{i,j,l}'
                    slj=f's_{l,j}'
                    sij=f's_{i,j}'
                    
                    xlj=f'x_{l,j}'
                    xij=f'x_{i,j}'
                    
                    size=self.factory.plants[j][1]
                    
                    p.addLinConstr(overlap, size)
                    
                    p.setSubMatrix(slj, overlap, 1)
                    p.setSubMatrix(sij, overlap, -1)
                    p.setSubMatrix(xij, overlap, M*np.eye(size,size))
                    p.setSubMatrix(xlj, overlap, M*np.eye(size,size))
                    
                    Belegung=self.Schedule[l].assignments[j]
                    p.setLinBound(overlap, -np.inf, 2*M-Belegung)
                    
                    
        p.addVar("C",1)
        
        for i in range(len(self.Schedule)):
            last_plant=len(self.factory.plants)-1
            
            cost=f'c_{i}'
            p.addLinConstr(cost, 1)
            p.setSubMatrix('C', cost, -1)
            p.setSubMatrix(f's_{i,last_plant}', cost, 1)
            Belgung=self.Schedule[i].assignments[last_plant]
            p.setLinBound(cost, -np.inf, -Belegung)
            
        p.addCost("C", 1)
        p.setupIndices()
        return p
            
hospital=Factory()
hospital.addPlant('MRT',2)
hospital.addPlant('Consultation',3)
consultation=Product("consult",hospital,[10,5])
checkup=Product("check",hospital,[3,7])

Terminplan=Scheduling(hospital)
Terminplan.addProduct(consultation)
Terminplan.addProduct(consultation)
Terminplan.addProduct(checkup)
Terminplan.addProduct(checkup)
Terminplan.addProduct(checkup)
Terminplan.addProduct(checkup)
Terminplan.addProduct(consultation)
Terminplan.addProduct(consultation)
Terminplan.addProduct(consultation)
Terminplan.addProduct(checkup)
Terminplan.addProduct(checkup)
Terminplan.addProduct(checkup)
Terminplan.addProduct(checkup)


p= Terminplan.createProblem()

cost, bounds , Aeq, beq, Aub, bub, integrality=p.createLinprog()


res=sp.optimize.linprog(cost, A_ub=Aub, b_ub=bub, A_eq=Aeq, b_eq=beq, bounds=bounds, method='highs', callback=None, options=None, x0=None, integrality=integrality)