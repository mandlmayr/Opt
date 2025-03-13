import numpy as np
from funs2 import jacdelta, fun1, jac2, delta, jac1, hess4, jacsigma, hess2, jac3, fun2, sigma, hess1, fun3

def constr(x):
    x1=x[0:10]
    x2=x[10:25]
    v=x[25:28]

    ret=np.zeros(7)

    ret[0:3]=sigma(x1, x2, 1)
    ret[3:7]=delta(x2, [1, 2])

    return ret

def JacConstr(x):
    x1=x[0:10]
    x2=x[10:25]
    v=x[25:28]

    jac=np.zeros((7,28))

    dx1_sigma, dx2_sigma=jacsigma(x1, x2, 1)
    jac[0:3,0:10]=dx1_sigma
    jac[0:3,10:25]=dx2_sigma

    dx2_delta=jacdelta(x2, [1, 2])
    jac[3:7,10:25]=dx2_delta

    return jac

def obj(x):
    x1=x[0:10]
    x2=x[10:25]
    v=x[25:28]

    obj=0

    obj+=fun1(x1, x2)
    obj+=fun2(x1, v)
    obj+=fun3(x1, v)

    return obj

def grad(x):
    x1=x[0:10]
    x2=x[10:25]
    v=x[25:28]

    grad=np.zeros(28)

    dx1_fun1, dx2_fun1=jac1(x1, x2, 1)
    grad[0:10]+=dx1_fun1(x1, x2, 1)
    grad[10:25]+=dx2_fun1(x1, x2, 1)

    dx1_fun2, dv_fun2=jac2(x1, v, 1)
    grad[0:10]+=dx1_fun2(x1, v, 1)
    grad[25:28]+=dv_fun2(x1, v, 1)

    dx1_fun3, dv_fun3=jac3(x1, v, 1)
    grad[0:10]+=dx1_fun3(x1, v, 1)
    grad[25:28]+=dv_fun3(x1, v, 1)

    return grad

def hess(x):
    x1=x[0:10]
    x2=x[10:25]
    v=x[25:28]

    hess=np.zeros((28,28))

    dx1x1_fun1, dx1x2_fun1, dx2x1_fun1, dx2x2_fun1=hess1(x1, x2, 1)
    hess[0:10,0:10]+=dx1x1_fun1
    hess[0:10,10:25]+=dx1x2_fun1
    hess[10:25,0:10]+=dx2x1_fun1
    hess[10:25,10:25]+=dx2x2_fun1

    dx1x1_fun2, dx1v_fun2, dvx1_fun2, dvv_fun2=hess2(x1, v, 1)
    hess[0:10,0:10]+=dx1x1_fun2
    hess[0:10,25:28]+=dx1v_fun2
    hess[25:28,0:10]+=dvx1_fun2
    hess[25:28,25:28]+=dvv_fun2

    dx1x1_fun3, dx1v_fun3, dvx1_fun3, dvv_fun3=hess4(x1, v, 1)
    hess[0:10,0:10]+=dx1x1_fun3
    hess[0:10,25:28]+=dx1v_fun3
    hess[25:28,0:10]+=dvx1_fun3
    hess[25:28,25:28]+=dvv_fun3

    return hess