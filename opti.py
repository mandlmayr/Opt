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
        
        self.matrix_blocks=[]
        self.lin_bound_blocks=[]
        self.bound_blocks=[]
        
        self.func_block=[]
        self.func_needed=set()
        
        self.nonlin_bound_blocks=[]
        
        

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
    
    def setSubMatrix(self, var_name, constr_name, matrix):
        if var_name not in self.variable_name:
            raise Exception("Variable does not exist!!!")
        if constr_name not in self.linear_constraint_name:
            raise Exception("Constraint does not exist!!!")
            
        indV=self.variable_name.index(var_name)
        indC=self.linear_constraint_name.index(constr_name)
        
        sizeV=self.variable_size[indV]
        sizeC=self.linear_constraint_size[indC]
            
        matC=matrix.shape[0]
        matV=matrix.shape[1]
        
        if matV != sizeV :
            raise Exception("Variable dimension is wrong!")
            
        if matC != sizeC :
            raise Exception("Constraint dimension is wrong!")
            
        self.matrix_blocks.append([constr_name,var_name,matrix])
        
    def createMatrixDense(self):
        self.matrix=np.zeros((self.ind_linear_constraint[-1],self.ind_var[-1]))
        
        for block in self.matrix_blocks:
            constr_name=block[0]
            var_name=block[1]
            mat=block[2]
            
            cStart, cEnd = self.indexLinConstr(constr_name)
            vStart, vEnd = self.indexVar(var_name)
            
            self.matrix[cStart:cEnd, vStart:vEnd]=mat
            
        
    def setLinBound(self, constr_name, lower, upper):
        if constr_name not in self.linear_constraint_name:
            raise Exception("Constraint does not exist!!!")
            
        indC=self.linear_constraint_name.index(constr_name)
        size=self.linear_constraint_size[indC]
        if len(lower)!=size:
            raise Exception("Lower dimension not correct!!!")
            
        if len(upper)!=size:
            raise Exception("Upper dimension not correct!!!")
            
        self.lin_bound_blocks.append([constr_name,lower,upper])
        
    def createLinBounds(self):
        self.lin_upper=np.zeros(self.ind_linear_constraint[-1])
        self.lin_lower=np.zeros(self.ind_linear_constraint[-1])
        
        for block in self.lin_bound_blocks:
            constr_name=block[0]
            lower=block[1]
            upper=block[2]
            
            cStart, cEnd = self.indexLinConstr(constr_name)
            
            self.lin_lower[cStart:cEnd]=lower
            self.lin_upper[cStart:cEnd]=upper
            
    def setBound(self, var_name, lower, upper):
        if var_name not in self.variable_name:
            raise Exception("Variable does not exist!!!")
            
        indV=self.variable_name.index(var_name)
        size=self.variable_size[indV]
        if len(lower)!=size:
            raise Exception("Lower dimension not correct!!!")
            
        if len(upper)!=size:
            raise Exception("Upper dimension not correct!!!")
            
        self.bound_blocks.append([var_name,lower,upper])        
        
    def CreateBounds(self):
        self.upper=np.zeros(self.ind_var[-1])
        self.lower=np.zeros(self.ind_var[-1])
        
        for block in self.bound_blocks:
            var_name=block[0]
            lower=block[1]
            upper=block[2]
            
            vStart, vEnd = self.indexVar(var_name)
            
            self.lower[vStart:vEnd]=lower
            self.upper[vStart:vEnd]=upper
            
    def setNonlinConstrFunc(self, constr_name,function_name, jac_name, variable_list, parameter):
        if constr_name not in self.nonlinear_constraint_name:
            raise Exception("Constraint does not exist!!!")
        if not (set(variable_list)<=set(self.variable_name)):
            raise Exception("Variable list is not contaiend in the problem variables!!!")         
        self.func_block.append([constr_name,function_name,jac_name,variable_list,parameter])
        self.func_needed.add(function_name)
        self.func_needed.add(jac_name)
    
    def setNonlinBound(self, constr_name, lower, upper):
        if constr_name not in self.nonlinear_constraint_name:
            raise Exception("Constraint does not exist!!!")
            
        indC=self.nonlinear_constraint_name.index(constr_name)
        size=self.nonlinear_constraint_size[indC]
        if len(lower)!=size:
            raise Exception("Lower dimension not correct!!!")
            
        if len(upper)!=size:
            raise Exception("Upper dimension not correct!!!")
            
        self.nonlinear_constraint_name.append([constr_name,lower,upper])

        
p1=Opt()
p1.addVar("x1", 10)
p1.addVar("x2",15)
p1.addLinConstr("bilanz1", 20)
p1.addLinConstr("bilanz2", 15)
p1.setupIndices()
A=np.ones((20,10))
B=np.ones((15,15))
p1.setSubMatrix("x1", "bilanz1", A)
p1.setSubMatrix("x2", "bilanz2", B)
p1.createMatrixDense()


p1.setLinBound("bilanz1", np.ones(20), np.ones(20))
p1.createLinBounds()
p1.setBound("x1", np.ones(10), np.ones(10))
p1.CreateBounds()

p1.addNonlinConstr("nonlin", 3)
p1.setNonlinConstrFunc("nonlin","sigma","jacsigma",["x1","x2"],1)
p1.setNonlinBound("nonlin", np.ones(3), np.ones(3))
