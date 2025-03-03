# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 20:32:00 2025

@author: Michael
"""

class Opt:
    def __init__(self):   
        self.variable_name=[]
        self.variable_size=[]
        
        self.linear_constraint_name=[]
        self.linear_constraint_size=[]
        
        self.nonlinear_constraint_name=[]
        self.nonlinear_constraint_size=[]
        

    def addVar(self, name, size):
        self.variable_name.append(name)
        self.variable_size.append(size)
        
    def addLinConstr(self, name, size):
        self.linear_constraint_name.append(name)
        self.linear_constraint_size.append(size)
        
    def addNonlinConstr(self,name, size):
        self.nonlinear_constraint_name.append(name)
        self.nonlinear_constraint_size.append(size)
        
