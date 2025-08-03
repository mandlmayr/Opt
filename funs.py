import numpy as np
from scipy.sparse import csr_matrix
from funs2 import jac4, fun3, hess4, fun4, hess2, jacdelta, fun2, hess1, jac1, jac3, hess5, jacsigma, jac2, sigma, delta, fun1
from params import s1, s2

def constrP(x ,s1 ,s2):
    x1=x[0:10]
    x2=x[10:25]
    v=x[25:28]

    ret=np.zeros(7)

    ret[0:3]=sigma(x1, x2, s1, s2)
    ret[3:7]=delta(x1, s1, s1)

    return ret

def JacConstrP(x ,s1 ,s2):
    x1=x[0:10]
    x2=x[10:25]
    v=x[25:28]

    jac=np.zeros((7,28))

    dx1_sigma, dx2_sigma=jacsigma(x1, x2, s1, s2)
    jac[0:3,0:10]=dx1_sigma
    jac[0:3,10:25]=dx2_sigma

    dx1_delta=jacdelta(x1, s1, s1)
    jac[3:7,0:10]=dx1_delta

    return csr_matrix(jac)

def objP(x ,s1 ,s2):
    x1=x[0:10]
    x2=x[10:25]
    v=x[25:28]

    obj=0

    obj+=fun1(x1, x2, s1, s1)
    obj+=fun2(x1, v, s1, s1)
    obj+=fun3(x1, v, s1, s1)
    obj+=fun4(x1, v, s1, s1)

    return obj

def gradP(x ,s1 ,s2):
    x1=x[0:10]
    x2=x[10:25]
    v=x[25:28]

    grad=np.zeros(28)

    dx1_fun1, dx2_fun1=jac1(x1, x2, s1, s1)
    grad[0:10]+=dx1_fun1(x1, x2, s1)
    grad[10:25]+=dx2_fun1(x1, x2, s1)

    dx1_fun2, dv_fun2=jac2(x1, v, s1, s1)
    grad[0:10]+=dx1_fun2(x1, v, s1)
    grad[25:28]+=dv_fun2(x1, v, s1)

    dx1_fun3, dv_fun3=jac3(x1, v, s1, s1)
    grad[0:10]+=dx1_fun3(x1, v, s1)
    grad[25:28]+=dv_fun3(x1, v, s1)

    dx1_fun4, dv_fun4=jac4(x1, v, s1, s1)
    grad[0:10]+=dx1_fun4(x1, v, s1)
    grad[25:28]+=dv_fun4(x1, v, s1)

    return grad

def hessP(x ,s1 ,s2):
    x1=x[0:10]
    x2=x[10:25]
    v=x[25:28]

    hess=np.zeros((28,28))

    dx1x1_fun1, dx1x2_fun1, dx2x1_fun1, dx2x2_fun1=hess1(x1, x2, s1, s1)
    hess[0:10,0:10]+=dx1x1_fun1
    hess[0:10,10:25]+=dx1x2_fun1
    hess[10:25,0:10]+=dx2x1_fun1
    hess[10:25,10:25]+=dx2x2_fun1

    dx1x1_fun2, dx1v_fun2, dvx1_fun2, dvv_fun2=hess2(x1, v, s1, s1)
    hess[0:10,0:10]+=dx1x1_fun2
    hess[0:10,25:28]+=dx1v_fun2
    hess[25:28,0:10]+=dvx1_fun2
    hess[25:28,25:28]+=dvv_fun2

    dx1x1_fun3, dx1v_fun3, dvx1_fun3, dvv_fun3=hess4(x1, v, s1, s1)
    hess[0:10,0:10]+=dx1x1_fun3
    hess[0:10,25:28]+=dx1v_fun3
    hess[25:28,0:10]+=dvx1_fun3
    hess[25:28,25:28]+=dvv_fun3

    dx1x1_fun4, dx1v_fun4, dvx1_fun4, dvv_fun4=hess5(x1, v, s1, s1)
    hess[0:10,0:10]+=dx1x1_fun4
    hess[0:10,25:28]+=dx1v_fun4
    hess[25:28,0:10]+=dvx1_fun4
    hess[25:28,25:28]+=dvv_fun4

    return csr_matrix(hess)

def constr(x):
    return constr(x, s1, s2)


def JacConstr(x):
    return JacConstrP(x, s1, s2)


def obj(x):
    return objP(x, s1, s2)


def grad(x):
    return gradP(x, s1, s2)


def hess(x):
    return hessP(x, s1, s2)
