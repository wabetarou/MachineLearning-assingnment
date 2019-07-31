import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cv
from mkdata import st_ops
from seven_cons import cons
from seven_adam import adam
from seven_adagrad import adagrad
from seven_rms import rms
from seven_adad import adad
from seven_nadam import nadam


x_1 = np.arange(-1.5, 3, 0.01)
x_2 = np.arange(-1.5, 3, 0.02)

X1, X2 = np.mgrid[-1.5:3:0.01, -1.5:3:0.02]
fValue = np.zeros((len(x_1), len(x_2)))

A = np.array([[250, 15],
              [ 15,  4]])
mu = np.array([[1],
               [2]])
lam = 0.89

for i in range(len(x_1)):
  for j in range(len(x_2)):
    inr = np.vstack([x_1[i], x_2[j]])
    fValue[i, j] = np.dot(np.dot((inr-mu).T, A), (inr- mu)) + lam * (np.abs(x_1[i]) + np.abs(x_2[j]))

# cvx
w_lasso = cv.Variable((2,1))
obj_fn = cv.quad_form(w_lasso - mu, A) +  lam * cv.norm(w_lasso, 1)
objective = cv.Minimize(obj_fn)
constraints = []
prob = cv.Problem(objective, constraints)
result = prob.solve(solver=cv.CVXOPT)
w_lasso = w_lasso.value

#plt.contour(X1, X2, fValue, 80)
cons = cons(A,mu,lam)
adam = adam(A,mu,lam)
adagrad = adagrad(A,mu,lam)
rms = rms(A,mu,lam)
adad = adad(A,mu,lam)
nadam = nadam(A,mu,lam)

minfvalue = np.dot(np.dot((w_lasso - mu).T, A), (w_lasso - mu)) + lam * np.sum(np.abs(w_lasso))
minOfMin = np.min([minfvalue,
                   np.min(adam[1]),
                   np.min(adagrad[1]),
                   np.min(rms[1]),
                   np.min(adad[1]),
                   np.min(nadam[1])])


#plt.semilogy(cons[1] - minOfMin, 'r-')
plt.semilogy(adam[1] - minOfMin, 'bs-', markersize=1, linewidth=0.5,label='adam',c='skyblue')
plt.semilogy(adagrad[1] - minOfMin, 'bs-', markersize=1, linewidth=0.5,label='adagrad',c='blue')
plt.semilogy(rms[1] - minOfMin, 'bs-', markersize=1, linewidth=0.5,label='rms',c='green')
plt.semilogy(adad[1] - minOfMin, 'bs-', markersize=1, linewidth=0.5,label='adad',c='brown')
plt.semilogy(nadam[1] - minOfMin, 'bs-', markersize=1, linewidth=0.5,label='nadam',c='black')
plt.yscale('log')
plt.legend()
plt.show()
