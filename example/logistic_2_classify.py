import numpy as np

import sys
sys.path.append('../sxlearn')
import logistic

data_path = '../data/'
if __name__ == '__main__':
    raw_input_data = np.loadtxt(data_path + 'test/2-classify.txt')
    x = raw_input_data[:, :2]
    y = raw_input_data[:, 2]

    lr = logistic.LogisticRegression2C(x,y)
    cost_list,theta_list = lr.train(500,0.001)
    print(lr.predict([[-0.026632,10.427743]]))
    print(lr.theta)
    bestx = np.arange(-3.0,3.0,0.1)
    besty = []
    for theta in theta_list[3:10]:
        by = -1*(theta[0]+theta[1]*bestx)/theta[2]
        besty.append(by)

    import matplotlib.pyplot as plt

    xcord1, ycord1, xcord2,ycord2 = [],[],[],[]
    for i in range(len(x)):
        if y[i] == 1:
            xcord1.append(x[i,0])
            ycord1.append(x[i,1])
        else:
            xcord2.append(x[i,0])
            ycord2.append(x[i,1])

    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=20, c='red', marker='s', alpha = .5)
    ax.scatter(xcord2, ycord2, s=20, c='green', alpha=.5)

    # ax.plot(bestx,besty)
    for by in besty:
        ax.plot(bestx, by)
        # print(by)

    plt.title('input point')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

    ax = np.linspace(1, 500, 500)
    plt.figure(1)
    plt.plot(ax, cost_list)
    plt.show()