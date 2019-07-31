import numpy as np
from mkdata import st_ops

def adagrad(A,mu,lam):
    x_init = np.array([[ 3],
                   [-1]])
    xt = x_init
    L = 1.01 * np.max(np.linalg.eig(2 * A)[0])
    eta0 = 500/L;

    x_history = []
    fvalues = []
    g_history = []
    delta = 0.02;
    for t in range(100):
      x_history.append(xt.T)
      grad = 2 * np.dot(A, xt-mu)

      g_history.append(grad.flatten().tolist())
      ht = np.sqrt(np.sum(np.array(g_history)**2, axis=0).T) + delta
      ht = ht.reshape(2,1)

      eta_t = eta0
      xth = xt - eta_t * (ht**-1 * grad)
      ht_inv = ht**-1
      xt = np.array([st_ops(xth[0], lam  * eta_t * ht_inv[0]),
                     st_ops(xth[1], lam  * eta_t * ht_inv[1])])

      fv = np.dot(np.dot((xt - mu).T, A), (xt - mu)) + lam * (np.abs(xt[0]) + np.abs(xt[1]))
      fvalues.append(fv)

    x_history = np.vstack(x_history)
    fvalues = np.vstack(fvalues)
    return x_history, fvalues
