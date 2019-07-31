# requirement
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cv
from two_method import pg

A = np.array([[3, 0.5],
              [0.5,   1]])
mu = np.array([[1],
               [2]])
lam = 2

x_1 = np.arange(-1.5, 3, 0.01)
x_2 = np.arange(-1.5, 3, 0.02)

X1, X2 = np.mgrid[-1.5:3:0.01, -1.5:3:0.02]
fValue = np.zeros((len(x_1), len(x_2)))

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

plt.contour(X1, X2, fValue)

lam_2 = pg(A,mu,2)
lam_4 = pg(A,mu,4)
lam_6 = pg(A,mu,6)
lam = pg(A,mu,10)

plt.plot(lam_2[0][:,0], lam_2[0][:,1], 'ro-', markersize=1, linewidth=0.5, color='red', label='lam=2')
plt.plot(lam_4[0][:,0], lam_4[0][:,1], 'ro-', markersize=1, linewidth=0.5, color='green', label = 'lam=4')
plt.plot(lam_6[0][:,0], lam_6[0][:,1], 'ro-', markersize=1, linewidth=0.5, color='blue', label='lam=6')
plt.plot(w_lasso[0], w_lasso[1], 'ko')

result_2 = np.linalg.norm(lam_2[0] - lam_2[0][-1],axis=1)
result_4 = np.linalg.norm(lam_4[0] - lam_4[0][-1],axis=1)
result_6 = np.linalg.norm(lam_6[0] - lam_6[0][-1],axis=1)
result = np.linalg.norm(lam[0] - lam[0][-1],axis=1)
#plt.plot([x for x in range(1,171)],result_2,'ro-', markersize=1, linewidth=0.5, color='red', label='lam=2')
#plt.plot([x for x in range(1,171)],result_4,'ro-', markersize=1, linewidth=0.5, color='green', label = 'lam=4')
#plt.plot([x for x in range(1,171)],result_6,'ro-', markersize=1, linewidth=0.5, color='blue', label='lam=6')
#plt.plot([x for x in range(1,171)],result,'ro-', markersize=1, linewidth=0.5, color='black', label='lam')
#plt.yscale('log')
plt.legend()
plt.show()
