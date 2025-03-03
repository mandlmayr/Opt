# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 20:32:00 2025

@author: Michael
"""
import numpy as np

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
        
    def setupIndices(self):
        self.ind_var=[0]
        self.ind_linear_constraint=[0]
        self.ind_nonlinear_constraint=[0]
        
        size_counter=0
        for size in self.variable_size:
            size_counter+=size
            self.ind_var.append(size_counter)
            
        size_counter=0
        for size in self.linear_constraint_size:
            size_counter+=size
            self.ind_linear_constraint.append(size_counter)
            
        size_counter=0
        for size in self.nonlinear_constraint_size:
            size_counter+=size
            self.ind_nonlinear_constraint.append(size_counter)
            
    def indexVar(self,var_name):
        ind=self.variable_name.index(var_name)
        return self.ind_var[ind], self.ind_var[ind+1]
        
    def indexLinConstr(self, constr_name):
        ind=self.linear_constraint_name.index(constr_name)
        return self.ind_linear_constraint[ind], self.ind_linear_constraint[ind+1]
    
    def indexNonlinConstr(self, constr_name):
        ind=self.nonlinear_constraint_name.index(constr_name)
        return self.ind_nonlinear_constraint[ind], self.ind_nonlinear_constraint[ind+1]
        
            
            
        
        
        
        
p1=Opt()
p1.addVar("x1", 10)
p1.addVar("x2",15)
p1.addLinConstr("bilanz1", 20)
p1.addLinConstr("bilanz2", 15)
p1.setupIndices()