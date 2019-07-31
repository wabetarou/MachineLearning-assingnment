import numpy as np
#import matplotlib.pyplot as plt
from mkdata import load_d4

def newton(x, y):
    lam = 1
    offset = np.ones((x.shape[0],1))#(200,1)
    x = np.hstack((x, offset))#(200,5)
    w = np.ones((3,x.shape[1]))#(3,5)
    loss_history = []
    while 1:
        loss = 0
        grad = np.zeros((3,x.shape[1]))
        hessian = [np.zeros((5,5)),np.zeros((5,5)),np.zeros((5,5))]
        z = x.dot(w.T)
        max = np.max(z,axis=1).reshape(200,1)
        max3 = np.hstack((np.hstack((max,max)),max))
        z -= max3
        softmax = np.exp(z)
        for softmaxi in softmax:
            softmaxi /= sum(softmaxi)
        for j in range(3):
            for xi,yi,softmaxi in zip(x,y,softmax):
                if j == yi:
                    loss -= np.log(softmaxi[j])
                    grad[j] += (softmaxi[j]-1)*xi
                else:
                    grad[j] += softmaxi[j]*xi
                xi = xi.reshape((1,5))
                hessian[j] += softmaxi[j]*(1-softmaxi[j]) * xi.T.dot(xi)
        loss /= x.shape[0]
        grad /= x.shape[0]
        for j in range(3):
            hessian[j] /= x.shape[0]
            hessian[j] += 2*lam*np.eye(5)
        loss += lam * np.linalg.norm(w)**2
        loss_history.append(loss)
        grad += 2*lam*w
        for j in range(3):
            d = - np.linalg.inv(hessian[j]).dot(grad[j])
            w[j] += d
        if np.linalg.norm(grad) < 1e-6:
            break
        print(loss)
    return loss_history
