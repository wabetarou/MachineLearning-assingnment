import numpy as np
from mkdata import load_d4
from mkdata import sgd_loss

def sgd_backtrack(lam, x, y):
    offset = np.ones((x.shape[0],1))
    x = np.hstack((x, offset))
    w = np.ones(x.shape[1])
    loss_history = []
    c = 0.5
    rho = 0.9
    while 1:
        loss,grad = sgd_loss(lam,w,x,y)
        if np.linalg.norm(grad) < 1e-6:
            break
        d = -grad
        a = 10
        while 1:
            if sgd_loss(lam,(w+a*d),x,y)[0]-loss < c*a*grad.dot(d):
                break
            a *= rho
        w = w + a*d
        print(loss)
        loss_history.append(loss)
    return loss_history
#plt.plot(np.arange(0,len(loss_history)),loss_history)
#plt.show()
