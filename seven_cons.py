import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cv
from mkdata import st_ops

def cons(A,mu,lam):
    x_init = np.array([[ 3],
                       [-1]])
    xt = x_init
    L = 1.01 * np.max(np.linalg.eig(2 * A)[0])

    x_historyPg = []
    fvaluesPg = []
    xt = x_init
    for t in range(100):
      x_historyPg.append(xt.T)
      grad = 2 * np.dot(A, xt-mu)
      xth = xt - 1/L * grad
      xt = st_ops(xth, lam * 1 / L)
      fv = np.dot(np.dot((xt - mu).T, A), (xt - mu)) + lam * (np.abs(xt[0]) + np.abs(xt[1]))
      fvaluesPg.append(fv)
    x_historyPg = np.vstack(x_historyPg)
    fvaluesPg = np.vstack(fvaluesPg)
    return x_historyPg,fvaluesPg
