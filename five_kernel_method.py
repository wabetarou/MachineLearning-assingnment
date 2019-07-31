import numpy as np
from mkdata import create_kernel
from mkdata import create_kernel_table
from mkdata import create_kernel_matrix


def kernel_method(x,y,a,lam):
    table = create_kernel_table(x,a)
    table += lam*np.eye(table.shape[0])
    w = np.linalg.inv(table).dot(y)
    return w

def kernel_method_d(x,y,a,lam,dtilde):
    n = x.shape[0]
    table = create_kernel_table(x,a)
    z = np.zeros(dtilde,dtype=int)
    for i in range(dtilde):
        z[i] += np.random.randint(n)
    newtable = np.zeros((n,n))
    for i in range(dtilde):
        if newtable[i][z[i]] == 0:
            newtable[i][z[i]] += table[i][z[i]]
            newtable[z[i]][i] += table[z[i]][i]
    newtable += lam*np.eye(n)
    w = np.linalg.inv(newtable).dot(y)
    return w

def display_result(x,y,a,w):
    f = 60
    testx = np.zeros((f**2,2))
    for i in range(f):
        for j in range (f):
            testx[i*f+j][0] = -1.5+i*3/f
            testx[i*f+j][1] = -1.5+j*3/f

    yhat = np.zeros(f**2)
    for i in range(f**2):
        yhat[i] += create_kernel(x,testx[i],a).dot(w)

    result = np.zeros(yhat.shape,dtype=int)
    for i in range(f**2):
        if yhat[i] > 0:
            result[i] += 1
        else:
            result[i] -= 1
    return testx,result
