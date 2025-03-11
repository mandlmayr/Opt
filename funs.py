import numpy as np
from funs2 import sigma, jacdelta, delta, jacsigma

def constr(x):
    x1=x[0:10]
    x2=x[10:25]

    ret=np.zeros(7)

    ret[0:3]=sigma(x1, x2, 1)
    ret[3:7]=delta(x2, [1, 2])

    return ret

def JacConstr(x):
    x1=x[0:10]

    x2=x[10:25]

    jac=np.zeros((7,25))

    dx1_sigma, dx2_sigma=jacsigma(x1, x2, 1)
    jac[0:3,0:10]=dx1_sigma
    jac[0:3,10:25]=dx2_sigma

    dx2_delta=jacdelta(x2, [1, 2])
    jac[3:7,10:25]=dx2_delta

    return jac