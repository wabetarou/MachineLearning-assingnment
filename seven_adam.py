import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cv
from mkdata import st_ops

def adam(A,mu,lam):
    x_init = np.array([[ 3],
                       [-1]])
    xt = x_init

    b1 = 0.7
    b2 = 0.99999
    ee = 1.0e-8
    aa= 0.2

    x_history = []
    fvalues = []
    g_history = []
    mm = np.zeros((2,1))
    vv = np.zeros((2,1))

    for t in range(1,101):
      x_history.append(xt.T)
      grad = 2 * np.dot(A, xt-mu)

      mm = b1 * mm + (1-b1) * grad
      vv = b2 * vv + (1 - b2) * (grad * grad)

      mmHat = mm / (1-b1**t)
      vvHat = vv / (1-b2**t)

      g_history.append(grad.T)

      rateProx = aa * np.ones((2, 1)) / (np.sqrt(vvHat) + ee)

      xth = xt -  mmHat * rateProx

      xt = np.array([st_ops(xth[0], lam  * rateProx[0]),
                     st_ops(xth[1], lam  * rateProx[1])])

      fv = np.dot(np.dot((xt - mu).T, A), (xt - mu)) + lam * (np.abs(xt[0]) + np.abs(xt[1]))
      fvalues.append(fv)

    x_history = np.vstack(x_history)
    fvalues = np.vstack(fvalues)
    return x_history, fvalues
