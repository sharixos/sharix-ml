import numpy as np

import sys
sys.path.append('../sxlearn')
import linear_model

data_path = '../data/'
if __name__ == '__main__':
    raw_input_data = np.loadtxt(data_path + 'house_price/ex1data2.txt', delimiter=',')
    x = raw_input_data[:, :2]
    y = raw_input_data[:, 2]

    lr = linear_model.LinearRegression(x, y, need_normalized=1)
    cost_list = lr.train(10000, 0.1)
    print(lr.predict([[1236,3]]))

    import matplotlib.pyplot as plt
    ax = np.linspace(1, 10000, 10000)
    plt.figure(1)
    plt.plot(ax, cost_list)
    plt.show()