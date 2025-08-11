# -*- coding: utf-8 -*-
"""
Created on Mon Aug 11 00:50:40 2025

@author: Michael
"""
from opti import Opt


class Factory:
    def __init__(self):
        self.plants=[]
        
    def addPlant(self, plant, amount):
        self.plants.append((plant,amount))
        
class Product:
    def __init__(self,factory, assignments):
        self.factory=factory
        self.assignments=assignments
        
        
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
        for product in self.Schedule:      
            for anlage in self.factory.plants:
                name = anlage[0]
                size= anlage[1]
                string="decision-"+name+f'{k}'
                p.addVar(string, size)
                p.addIntegrality(string, 1)
                p.setBound(string, 0, 1)
                
                p.addVar("startingTime-"+name+f'{k}',1)
            k=k+1
        p.addVar("DurchlaufZeit",1)
        p.setupIndices()
        return p
            
hospital=Factory()
hospital.addPlant('MRT',1)
hospital.addPlant('Consultation',2)

consultation=Product(hospital,[1,1])
checkup=Product(hospital,[0,1])

Terminplan=Scheduling(hospital)
Terminplan.addProduct(consultation)
Terminplan.addProduct(consultation)
Terminplan.addProduct(checkup)

p= Terminplan.createProblem()