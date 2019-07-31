import matplotlib.pyplot as plt
import numpy as np
from one_mult_sgd import sgd
from one_mult_newton import newton
from mkdata import load_d5


x, y = load_d5()
sgd_history = sgd(x,y)
print('start newton')
newton_history = newton(x,y)

min = min(newton_history[-1],sgd_history[-1])
sgd_history -= min
newton_history -= min

plt.plot(np.arange(0,len(sgd_history)),sgd_history,label='sgd')
plt.plot(np.arange(0,len(newton_history)),newton_history,label='newton')
plt.legend()
plt.show()
