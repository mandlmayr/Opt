# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 20:32:00 2025

@author: Michael
"""
import numpy as np
from scipy.sparse import  csr_matrix, lil_matrix

class Opt:
    def __init__(self,sparse=False):   
        self.variable_name=[]
        self.variable_size=[]
        
        self.parameter_name=[]
        
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
        
        self.obj_func_block=[]
        
        self.sparsity=sparse

    def addVar(self, name,size):
        self.variable_name.append(name)
        self.variable_size.append(size)
    
    def addPar(self, name):
        self.parameter_name.append(name)
        
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
        
    def createMatrix(self):
        self.matrix=lil_matrix((self.ind_linear_constraint[-1],self.ind_var[-1]))
        
        for block in self.matrix_blocks:
            constr_name=block[0]
            var_name=block[1]
            mat=block[2]
            
            cStart, cEnd = self.indexLinConstr(constr_name)
            vStart, vEnd = self.indexVar(var_name)
            
            self.matrix[cStart:cEnd, vStart:vEnd]=mat
            
        if(self.sparsity):
            self.matrix=csr_matrix(self.matrix)
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
            
    def setNonlinConstrFunc(self, constr_name,function_name, jac_name, variable_list, parameter_list):
        if constr_name not in self.nonlinear_constraint_name:
            raise Exception("Constraint does not exist!!!")
        if not (set(variable_list)<=set(self.variable_name)):
            raise Exception("Variable list is not contaiend in the problem variables!!!")     
        if not (set(parameter_list)<=set(self.parameter_name)):
            raise Exception("Variable list is not contaiend in the problem variables!!!")         
        self.func_block.append([constr_name,function_name,jac_name,variable_list,parameter_list])
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
            
    def createNonlinFunctions(self,filename_dest, filename_funcs, filename_parameters):
        space="    "
        linebreak="\n"
        string="import numpy as np"+linebreak
        if(self.sparsity):
            string+="from scipy.sparse import csr_matrix, lil_matrix"+linebreak
        string+="from "+ filename_funcs+" import "

        first=0
        for func in self.func_needed:
            if first==0:
                first=1
                string+=func
            else:
                string+=", "+func
        string+=linebreak
        string+="from "+ filename_parameters+" import "
        first=0
        for parameter in self.parameter_name:
            if first==0:
                first=1
                string+=parameter
            else:
                string+=", "+parameter
            
        string+=linebreak+linebreak
        
        string+= "def constrP(x"
        for par in self.parameter_name:
            string+=" ,"+par
        
        string+="):"+linebreak
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
            parameters=block[4]
            
            cStart, cEnd = self.indexNonlinConstr(constr)
            
            string+=space+"ret["+str(cStart)+":"+str(cEnd)+"]="+func+"("
            first=0
            for var in variables:
                if first==0:
                    first=1
                    string+=var
                else:
                    string+=", "+var       
                    
            for parameter in parameters:
                string+=", "+parameter
            string +=")"+linebreak
        string+=linebreak
        string+=space+"return ret"
        
        string+=linebreak+linebreak
        string+="def JacConstrP(x"
        for par in self.parameter_name:
            string+=" ,"+par      
        string+="):"+linebreak
        for var in self.variable_name:
            vstart, vend=self.indexVar(var)
            string+=space+var+"=x["+str(vstart)+":"+str(vend)+"]"+linebreak
        string+=linebreak
        string+=space+"jac=lil_matrix(("+str(self.ind_nonlinear_constraint[-1])+","+str(self.ind_var[-1])+"))"+linebreak+linebreak
            
        for block in self.func_block:
            constr=block[0]
            func=block[1]
            jac=block[2]
            variables=block[3]
            parameters=block[4]
            
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
            for parameter in parameters:
                string+=", "+parameter
            string +=")"+linebreak
            for var in variables:
                cStart, cEnd=self.indexNonlinConstr(constr)
                vStart, vEnd=self.indexVar(var)
                string+=space+"jac["+str(cStart)+":"+str(cEnd)+","+str(vStart)+":"+str(vEnd)+"]=d"+var+"_"+func+linebreak
            string+=linebreak
        string+=space+"return csr_matrix(jac)"
        string+=linebreak+linebreak
        
        
        
        
        string+="def objP(x"   
        for par in self.parameter_name:
            string+=" ,"+par             
        string+="):"+linebreak
        for var in self.variable_name:
            vstart, vend=self.indexVar(var)
            string+=space+var+"=x["+str(vstart)+":"+str(vend)+"]"+linebreak
        string+=linebreak
        string+=space+"obj=0"+linebreak+linebreak
        for block in self.obj_func_block:
            func=block[0]
            jac=block[1]
            hess=block[2]            
            variables=block[3]
            parameter=block[4]
            
            string+=space+"obj+="+func+"("
            first=0
            for var in variables:
                if first==0:
                    first=1
                    string+=var
                else:
                    string+=", "+var     
            for parameter in parameters:
                string+=", "+parameter
            string +=")"+linebreak
        string+=linebreak
        string+=space+"return obj"
        string+=linebreak+linebreak
        
        
        string+="def gradP(x"
        for par in self.parameter_name:
            string+=" ,"+par             
        string+="):"+linebreak
        for var in self.variable_name:
            vstart, vend=self.indexVar(var)
            string+=space+var+"=x["+str(vstart)+":"+str(vend)+"]"+linebreak
        string+=linebreak
        string+=space+"grad=np.zeros("+str(self.ind_var[-1])+")"+linebreak+linebreak
        for block in self.obj_func_block:
            func=block[0]
            jac=block[1]
            hess=block[2]            
            variables=block[3]
            parameter=block[4]
            
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
            for parameter in parameters:
                string+=", "+parameter
            string +=")"+linebreak
            for var in variables:
                vStart, vEnd = self.indexVar(var)
                
                string+=space+"grad["+str(vStart)+":"+str(vEnd)+"]+="+"d"+var+"_"+func+"("
                first=0
                for var in variables:
                    if first==0:
                        first=1
                        string+=var
                    else:
                        string+=", "+var         
                string+=", "+str(parameter)+")"+linebreak
            string+=linebreak
            
        string+=space+"return grad"+linebreak+linebreak
        
        
        
        
        string+="def hessP(x"
        for par in self.parameter_name:
            string+=" ,"+par             
        string+="):"+linebreak
        for var in self.variable_name:
            vstart, vend=self.indexVar(var)
            string+=space+var+"=x["+str(vstart)+":"+str(vend)+"]"+linebreak
        string+=linebreak
        string+=space+"hess=lil_matrix(("+str(self.ind_var[-1])+","+str(self.ind_var[-1])+"))"+linebreak+linebreak
        for block in self.obj_func_block:
            func=block[0]
            jac=block[1]
            hess=block[2]            
            variables=block[3]
            parameter=block[4]
            
            string+=space
            first=0
            for var in variables:
                for var2 in variables:
                    if first==0:
                        first=1
                        string+="d"+var+var2+"_"+func
                    else:
                        string+=", d"+var+var2+"_"+func
            string+="="+hess+"("
            first=0
            for var in variables:
                if first==0:
                    first=1
                    string+=var
                else:
                    string+=", "+var         
            for parameter in parameters:
                string+=", "+parameter
            string +=")"+linebreak
            
            
            for var in variables:
                for var2 in variables:
                    vStart, vEnd = self.indexVar(var) 
                    v2Start,v2End= self.indexVar(var2)
                    
                    string+=space+"hess["+str(vStart)+":"+str(vEnd)+"," +str(v2Start)+":"+str(v2End)+"]+="+"d"+var+var2+"_"+func+linebreak
            string+=linebreak
        string+=linebreak
        string+=space+"return csr_matrix(hess)"
                    
        string+=linebreak+linebreak
        string+="def constr(x):"+linebreak
        string+=space+"return constr(x"
        for par in self.parameter_name:
            string+=", "+par             
        string+=")"+linebreak
        
        string+=linebreak+linebreak
        string+="def JacConstr(x):"+linebreak
        string+=space+"return JacConstrP(x"
        for par in self.parameter_name:
            string+=", "+par             
        string+=")"+linebreak
        
        string+=linebreak+linebreak
        string+="def obj(x):"+linebreak
        string+=space+"return objP(x"
        for par in self.parameter_name:
            string+=", "+par             
        string+=")"+linebreak
    
        string+=linebreak+linebreak
        string+="def grad(x):"+linebreak
        string+=space+"return gradP(x"
        for par in self.parameter_name:
            string+=", "+par             
        string+=")"+linebreak
        
        string+=linebreak+linebreak
        string+="def hess(x):"+linebreak
        string+=space+"return hessP(x"
        for par in self.parameter_name:
            string+=", "+par             
        string+=")"+linebreak
        
        f = open(filename_dest+".py", "w")
        f.write(string)
        f.close()
        
    def addObjPart(self, function_name, jac_name, hess_name, variable_list, parameter):
        self.obj_func_block.append([function_name,jac_name,hess_name,variable_list,parameter])
        self.func_needed.add(function_name)
        self.func_needed.add(jac_name)
        self.func_needed.add(hess_name)
        
        
    def createNonlinearProgramm(self):
        self.createBounds()
        self.createLinBounds()
        self.createMatrix()
        self.createNonlinBounds()
        
        return self.lower, self.upper, self.matrix, self.lin_lower, self.lin_upper, self.nonlin_lower, self.nonlin_upper
        
      
        


p1=Opt(True)


p1.addVar("x1", 10)
p1.addVar("x2",15)
p1.addVar("v",3)


p1.addPar('s1')
p1.addPar('s2')

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

p1.setNonlinConstrFunc("nonlin","sigma","jacsigma",["x1","x2"],["s1","s2"])
p1.setNonlinConstrFunc("nonlin2", "delta", "jacdelta", ["x1"], ['s1','s1'])
p1.setNonlinBound("nonlin", np.ones(3), np.ones(3))


p1.addObjPart("fun1", "jac1", "hess1", ["x1","x2"], ["s1"])
p1.addObjPart("fun2", "jac2", "hess2", ["x1","v"], ['s1','s1'])
p1.addObjPart("fun3", "jac3", "hess4", ["x1","v"], ['s1','s1'])
p1.addObjPart("fun4", "jac4", "hess5", ["x1","v"], ['s1','s1'])


p1.createNonlinFunctions("funs", "funs2","params")

a,b,c,d,e,f,g =p1.createNonlinearProgramm()


