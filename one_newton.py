import numpy as np
#import matplotlib.pyplot as plt
from mkdata import load_d4

def newton(lam, x, y):
    offset = np.ones((x.shape[0],1))
    x = np.hstack((x, offset))
    w = np.ones(x.shape[1])
    loss_history = []
    while 1:
        loss = 0
        grad = 0
        hessian = 0
        for xi,yi in zip(x,y):#データサイズごと
            exp = np.exp(-xi.dot(w)*yi)
            loss += np.log(1+exp)
            grad += -exp*yi*xi/(1+exp)
            xi = xi.reshape((1,-1))
            hessian += exp/(1+exp)**2 * np.dot(xi.T,xi) * yi**2
        loss /= x.shape[0]
        grad /= x.shape[0]
        hessian /= x.shape[0]
        loss += lam * w.dot(w)
        grad += 2*lam*w
        hessian += 2*lam*np.eye(5)
        d = - np.linalg.inv(hessian).dot(grad)
        w = w + d
        loss_history.append(loss)
        if np.linalg.norm(grad) < 1e-6:
            break
        print(loss)
    return loss_history
