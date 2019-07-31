import numpy as np
from mkdata import st_ops

def pg(A,mu,lam):
    x_init = np.array([[ 3],
                       [-1]])
    L = 1.01 * np.max(np.linalg.eig(2 * A)[0])

    x_history = []
    fvalues = []
    xt = x_init
    for t in range(170):
      x_history.append(xt.T)
      grad = 2 * np.dot(A, xt-mu)
      xth = xt - 1/L * grad
      xt = st_ops(xth, lam * 1 / L)
      fv = np.dot(np.dot((xt - mu).T, A), (xt - mu)) + lam * (np.abs(xt[0]) + np.abs(xt[1]))
      fvalues.append(fv)

    x_history = np.vstack(x_history)
    fvalues = np.vstack(fvalues)
    return x_history,fvalues
    
