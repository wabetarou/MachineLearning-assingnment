import numpy as np
from mkdata import st_ops

def adad(A,mu,lam):
    x_init = np.array([[ 3],
                   [-1]])
    xt = x_init

    x_history = []
    fvalues = []
    g_history = []
    beta = 0.99
    epsilon = 5e-3
    st = 0
    ht = 0
    for t in range(100):
        x_history.append(xt.T)
        grad = 2 * np.dot(A, xt-mu)
        g_history = (np.array(g_history)*beta).tolist()
        g_history.append(grad.flatten().tolist())

        ht = np.sum(np.array(g_history)**2*(1-beta), axis=0).T
        ht = ht.reshape(2,1)


        vt = np.sqrt((st + epsilon)/(ht + epsilon)) * grad


        xth = xt - vt

        xt = np.array([st_ops(xth[0], (np.sqrt((st + epsilon)/(ht + epsilon)) * lam)[0]),
                     st_ops(xth[1], (np.sqrt((st + epsilon)/(ht + epsilon)) * lam)[1])])
        st = beta * st + (1-beta) * vt**2

        fv = np.dot(np.dot((xt - mu).T, A), (xt - mu)) + lam * (np.abs(xt[0]) + np.abs(xt[1]))
        fvalues.append(fv)
    x_history = np.vstack(x_history)
    fvalues = np.vstack(fvalues)
    return x_history, fvalues
