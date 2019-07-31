import numpy as np
from mkdata import load_d4
from mkdata import sgd_loss

def sgd(lam, x, y, a):
    offset = np.ones((x.shape[0],1))
    x = np.hstack((x, offset))
    w = np.ones(x.shape[1])
    loss_history = []
    while 1:
        loss,grad = sgd_loss(lam,w,x,y)
        if np.linalg.norm(grad) < 1e-6:
            break
        d = -grad
        w = w + a*d
        print(loss)
        loss_history.append(loss)
    return loss_history
#plt.plot(np.arange(0,len(loss_history)),loss_history)
#plt.show()
