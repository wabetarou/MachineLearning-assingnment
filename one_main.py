import matplotlib.pyplot as plt
import numpy as np
from one_sgd import sgd
from one_sgd_backtrack import sgd_backtrack
from one_newton import newton
from mkdata import load_d4

lam = 1
x, y = load_d4()
print('start sgd01')
sgd_history_01 = sgd(lam,x,y,0.4)
print('start sgd05')
sgd_history_05 = sgd(lam,x,y,0.5)
print('start sgd_backtrack')
sgd_backtrack_history = sgd_backtrack(lam,x,y)
print('start newton')
newton_history = newton(lam,x,y)
min = min(sgd_history_01[-1],sgd_history_05[-1],sgd_backtrack_history[-1],newton_history[-1])
sgd_history_01 -= min
sgd_history_05 -= min
sgd_backtrack_history -= min
newton_history -= min
plt.plot(np.arange(0,len(sgd_history_01)),sgd_history_01,linestyle='dashdot',label='sgd_01')
plt.plot(np.arange(0,len(sgd_history_05)),sgd_history_05,linestyle='dashdot',label='sgd_05')
plt.plot(np.arange(0,len(sgd_backtrack_history)),sgd_backtrack_history,linestyle='solid',label='sgd_backtrack')
plt.plot(np.arange(0,len(newton_history)),newton_history,linestyle='dashed',label='newton')

#ax = plt.gca()
#ax.spines['top'].set_color('none')
#ax.set_yscale('log')
plt.yscale('log')
plt.legend()
plt.show()
