import numpy as np

import sys
sys.path.append('../sxlearn')
import linear_model

data_path = '../data/'
if __name__ == '__main__':
    raw_input_data = np.loadtxt(data_path + 'test/y_x_label.txt', delimiter=',')
    x = raw_input_data[:, :2]
    y = raw_input_data[:, 2]

    sensor = linear_model.Sensor(x,y,need_normalized=1)
    cost_list = sensor.train(100,0.01)
    print(sensor.predict([[3.4,4.2]]))

    import matplotlib.pyplot as plt
    ax = np.linspace(1, 100, 100)
    plt.figure(1)
    plt.plot(ax, cost_list)
    plt.show()