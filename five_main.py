import matplotlib.pyplot as plt
import numpy as np
from mkdata import load_toy1
from five_kernel_method import kernel_method
from five_kernel_method import kernel_method_d
from five_kernel_method import display_result

lam = 1
a=5
n = 200
x_200, y_200 = load_toy1(n)
x = x_200[:int(n/2)]
y = y_200[:int(n/2)]
x_50 = x_200[:int(n/4)]
y_50 = y_200[:int(n/4)]
n = x.shape[0]
dtilde = n-1

w = kernel_method(x,y,a,lam)
w_200 = kernel_method(x_200,y_200,a,lam)
w_50 = kernel_method(x_50,y_50,a,lam)
w_n = kernel_method(x[:int(n/2)],y[:int(n/2)],a,lam)
w_d = kernel_method_d(x,y,a,lam,dtilde)
w_m = kernel_method(x,y,a*1e-1,lam)
w_p = kernel_method(x,y,a*1e1,lam)

result_50_x, result_50_y = display_result(x_50,y,a,w_50)
result_x, result_y = display_result(x,y,a,w)
result_200_x, result_200_y = display_result(x_200,y,a,w_200)

result_x_d,result_y_d = display_result(x,y,a,w_d)

result_x_m,result_y_m = display_result(x,y,a*1e-1,w_m)
result_x_p,result_y_p = display_result(x,y,a*1e1,w_p)

fig, axes = plt.subplots(nrows=3,ncols=4)


#pltR.scatter(x[:,0],x[:,1],c=y)
#pltR.scatter(testx[:,0],testx[:,1],c=testy,marker='.')

axes[0][0].scatter(x[:,0],x[:,1],c=y)
axes[0][1].scatter(result_50_x[:,0],result_50_x[:,1],c=result_50_y,marker='.')
axes[0][2].scatter(result_x[:,0],result_x[:,1],c=result_y,marker='.')
axes[0][3].scatter(result_200_x[:,0],result_200_x[:,1],c=result_200_y,marker='.')

axes[1][0].scatter(result_x[:,0],result_x[:,1],c=result_y,marker='.')
axes[1][1].scatter(result_x_d[:,0],result_x_d[:,1],c=result_y_d,marker='.')

axes[2][0].scatter(result_x[:,0],result_x[:,1],c=result_y,marker='.')
axes[2][1].scatter(result_x_m[:,0],result_x_m[:,1],c=result_y_m,marker='.')
axes[2][2].scatter(result_x_p[:,0],result_x_p[:,1],c=result_y_p,marker='.')

plt.show()
