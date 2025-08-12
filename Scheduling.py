# -*- coding: utf-8 -*-
"""
Created on Mon Aug 11 00:50:40 2025

@author: Michael
"""
import os
os.environ["OMP_NUM_THREADS"] = "20"  # use 4 threads
from opti import Opt
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

M=1000
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
                    
                
            for l in range(0, i ):
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
        
        for i in range(0,len(self.Schedule)):
            last_plant=len(self.factory.plants)-1
            
            cost=f'c_{i}'
            p.addLinConstr(cost, 1)
            p.setSubMatrix('C', cost, -1)
            p.setSubMatrix(f's_{i,last_plant}', cost, 1)
            Belgung=self.Schedule[i].assignments[last_plant]
            p.setLinBound(cost, -np.inf, -Belegung)
            
        p.addCost("C", 1)
        p.setupIndices()
        
        self.p=p
        
    def optimize(self):
        self.createProblem()
        cost, bounds , Aeq, beq, Aub, bub, integrality=self.p.createLinprog()
        
        highs_options = {
            'time_limit': 10,
            'disp':True,
            'presolve':True
         }
        
        res=sp.optimize.linprog(cost, A_ub=Aub, b_ub=bub, A_eq=Aeq, b_eq=beq, bounds=bounds,options=highs_options, method='highs' ,callback=None, integrality=integrality)
        self.result=res.x
    
    def getSchedule(self,i):
        s=[]
        x=[]
        for j in range(len(self.factory.plants)):
            sj=self.p.extractVar(f's_{i,j}', self.result)
            
            xj=self.p.extractVar(f'x_{i,j}', self.result)
            
            print(self.factory.plants[j][0],sj)
            s.append(sj)
            x.append(xj)
        return s,x
    
    def printGantt(self):
        data=[]
        for k in range(len(self.Schedule)):
            product=self.Schedule[k]
            name=product.name
            s,x= self.getSchedule(k)
            
            for j in range(len(self.factory.plants)):
                sj=s[j][0]
                xj=x[j]
                machine=self.factory.plants[j][0]
                number=np.where(xj >= 0.9)[0][0]
                duration=product.assignments[j]
                
                
                data.append({                  "ProductName": name,
                                               "ProductIndex" : k,
                                               "plant": machine,
                                               "PlantIndex": number,
                                               "start": sj,
                                               "duration": duration
                                           })
            

        # Unique plants (machines) for Y-axis
        plants = sorted(set(f"{d['plant']}{d['PlantIndex']}" for d in data))
        plant_to_y = {p: i for i, p in enumerate(plants)} 

        # Unique products for colors
        products = sorted(set(f"{d['ProductName']}_{d['ProductIndex']}" for d in data))
        colors = plt.cm.get_cmap('tab20', len(products))
        product_to_color = {prod: colors(i) for i, prod in enumerate(products)}

        fig, ax = plt.subplots(figsize=(12, 6))

        for d in data:
            plant_label = f"{d['plant']}{d['PlantIndex']}"
            y = plant_to_y[plant_label]
            product_label = f"{d['ProductName']}_{d['ProductIndex']}"
            start = d['start']
            duration = d['duration']

            ax.barh(y=y, width=duration, left=start, height=0.8, color=product_to_color[product_label], edgecolor='black')

        # Y-axis ticks and labels
        ax.set_yticks(list(plant_to_y.values()))
        ax.set_yticklabels(list(plant_to_y.keys()))

        ax.set_xlabel('Time')
        ax.set_title('Gantt Chart: Product Scheduling on Machines')

        # Create legend for products
        handles = [plt.Rectangle((0,0),1,1, color=product_to_color[prod]) for prod in products]
        ax.legend(handles, products, title="Products")

        plt.tight_layout()
        plt.show()

        
hospital=Factory()
hospital.addPlant('MRT',2)
hospital.addPlant('Consultation',3)
hospital.addPlant('BloodTesting',2)
hospital.addPlant('release', 2)

consultation=Product("consult",hospital,[10,5,0,1])
checkup=Product("check",hospital,[3,7,20,1])

Terminplan=Scheduling(hospital)
Terminplan.addProduct(consultation)
Terminplan.addProduct(consultation)
Terminplan.addProduct(checkup)
Terminplan.addProduct(checkup)
Terminplan.addProduct(checkup)
Terminplan.addProduct(checkup)
Terminplan.addProduct(consultation)
Terminplan.addProduct(checkup)
Terminplan.addProduct(checkup)
Terminplan.addProduct(checkup)
Terminplan.addProduct(checkup)
Terminplan.addProduct(checkup)
Terminplan.addProduct(consultation)
Terminplan.addProduct(checkup)
Terminplan.addProduct(checkup)
Terminplan.addProduct(checkup)
Terminplan.addProduct(checkup)
Terminplan.addProduct(checkup)
Terminplan.addProduct(consultation)
Terminplan.addProduct(checkup)
Terminplan.addProduct(checkup)
Terminplan.addProduct(checkup)

Terminplan.optimize()

Terminplan.getSchedule(5)
a=Terminplan.printGantt()