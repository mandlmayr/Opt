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
        
    def createBounds(self):
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
            
        self.nonlin_bound_blocks.append([constr_name,lower,upper])

    def createNonlinBounds(self):
        self.nonlin_upper=np.zeros(self.ind_nonlinear_constraint[-1])
        self.nonlin_lower=np.zeros(self.ind_nonlinear_constraint[-1])
        
        for block in self.nonlin_bound_blocks:
            constr_name=block[0]
            lower=block[1]
            upper=block[2]
            
            cStart, cEnd = self.indexNonlinConstr(constr_name)
            
            self.nonlin_lower[cStart:cEnd]=lower
            self.nonlin_upper[cStart:cEnd]=upper
            
    def createNonlinFunctions(self,filename_dest, filename_funcs):
        space="    "
        linebreak="\n"
        string="import numpy as np"+linebreak
        string+="from "+ filename_funcs+" import "
        first=0
        for func in self.func_needed:
            if first==0:
                first=1
                string+=func
            else:
                string+=", "+func
            
            
        string+=linebreak+linebreak
        
        string+= "def constr(x):"+linebreak
        for var in self.variable_name:
            vstart, vend=self.indexVar(var)
            string+=space+var+"=x["+str(vstart)+":"+str(vend)+"]"+linebreak
        string+=linebreak
        string+=space+"ret=np.zeros("+str(self.ind_nonlinear_constraint[-1])+")"+linebreak+linebreak
        
        
        for block in self.func_block:
            constr=block[0]
            func=block[1]
            jac=block[2]
            variables=block[3]
            parameter=block[4]
            
            cStart, cEnd = self.indexNonlinConstr(constr)
            
            string+=space+"ret["+str(cStart)+":"+str(cEnd)+"]="+func+"("
            first=0
            for var in variables:
                if first==0:
                    first=1
                    string+=var
                else:
                    string+=", "+var         
            string+=", "+str(parameter)+")"+linebreak
        string+=linebreak
        string+=space+"return ret"
        
        string+=linebreak+linebreak
        string+="def JacConstr(x):"+linebreak
        for var in self.variable_name:
            vstart, vend=self.indexVar(var)
            string+=space+var+"=x["+str(vstart)+":"+str(vend)+"]"+linebreak+linebreak
        string+=space+"jac=np.zeros(("+str(self.ind_nonlinear_constraint[-1])+","+str(self.ind_var[-1])+"))"+linebreak+linebreak
            
        for block in self.func_block:
            constr=block[0]
            func=block[1]
            jac=block[2]
            variables=block[3]
            parameter=block[4]
            
            cStart, cEnd = self.indexNonlinConstr(constr)
            string+=space
            first=0
            for var in variables:
                if first==0:
                    first=1
                    string+="d"+var+"_"+func
                else:
                    string+=", "+"d"+var+"_"+func         
            string+="="+jac+"("
            first=0
            for var in variables:
                if first==0:
                    first=1
                    string+=var
                else:
                    string+=", "+var         
            string+=", "+str(parameter)+")"+linebreak
            for var in variables:
                cStart, cEnd=self.indexNonlinConstr(constr)
                vStart, vEnd=self.indexVar(var)
                string+=space+"jac["+str(cStart)+":"+str(cEnd)+","+str(vStart)+":"+str(vEnd)+"]=d"+var+"_"+func+linebreak
            string+=linebreak
        string+=space+"return jac"
                    
        f = open(filename_dest+".py", "w")
        f.write(string)
        f.close()

            
p1=Opt()


p1.addVar("x1", 10)
p1.addVar("x2",15)

p1.addLinConstr("bilanz1", 20)
p1.addLinConstr("bilanz2", 15)        

p1.addNonlinConstr("nonlin", 3)
p1.addNonlinConstr("nonlin2", 4)

p1.setupIndices()

A=np.ones((20,10))
B=np.ones((15,15))
p1.setSubMatrix("x1", "bilanz1", A)
p1.setSubMatrix("x2", "bilanz2", B)

p1.setLinBound("bilanz1", np.ones(20), np.ones(20))

p1.setBound("x1", np.ones(10), np.ones(10))

p1.setNonlinConstrFunc("nonlin","sigma","jacsigma",["x1","x2"],1)
p1.setNonlinConstrFunc("nonlin2", "delta", "jacdelta", ["x2"], [1,2])
p1.setNonlinBound("nonlin", np.ones(3), np.ones(3))




p1.createBounds()
p1.createLinBounds()
p1.createMatrixDense()
p1.createNonlinBounds()
p1.createNonlinFunctions("funs", "funs2")


