import numpy as np
from mkdata import load_d4
from mkdata import sgd_mult_loss

def sgd(x, y):
    lam = 1
    offset = np.ones((x.shape[0],1))
    x = np.hstack((x, offset))
    w = np.ones((3,x.shape[1]))
    loss_history = []
    c = 0.5
    rho = 0.9
    while 1:
        loss, grad = sgd_mult_loss(lam,w,x,y)
        if np.linalg.norm(grad) < 1e-6:
            break
        d = -grad
        a = 0.05
        #while 1:
            #if sgd_mult_loss(lam,(w+a*d),x,y)[0]-loss < c*a*np.trace(grad.dot(d.T)):
            #    break
            #a *= rho
        loss_history.append(loss)
        w = w + a*d
        print(loss)
    return loss_history
#plt.plot(np.arange(0,len(loss_history)),loss_history)
#plt.show()
