from sklearn import datasets
# from sklearn.model_selection import cross_val_predict
# from sklearn import linear_model
import matplotlib.pyplot as plt
import numpy as np 

import sys
sys.path.append('../sxlearn')
import linear_model



boston = datasets.load_boston()
x = boston.data
y = boston.target
lr = linear_model.LinearRegression(x,y,need_normalized = 1)
cost_list = lr.train(10000,0.1)


import matplotlib.pyplot as plt
ax = np.linspace(1, 10000, 10000)
plt.figure(1)
plt.plot(ax, cost_list)
plt.show()


