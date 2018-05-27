import numpy as np
import math
from functools import reduce


def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))


class NN_Logistic(object):
    """ single layer neural network using sigmoid as activation

    Author: [@Wang Xin Gang](https://github.com/sharixos)

    Website: http://www.sharix.site/

    Github: https://github.com/sharixos

    """

    def __init__(self, x_dim, labels):
        y_dim = len(labels)
        self.x_dim, self.y_dim = x_dim, y_dim
        weights_scale = math.sqrt(x_dim*y_dim)
        # self.weights = np.random.standard_normal((x_dim, y_dim)) / weights_scale
        # self.bias = np.random.standard_normal(y_dim)
        self.weights = np.ones((x_dim, y_dim))
        self.bias = np.zeros(y_dim)

        self.gratitude_w = np.zeros(self.weights.shape)
        self.gratitude_b = np.zeros(self.bias.shape)

        y_eyes = np.eye(y_dim)
        self.label2vec = {}
        for i in range(y_dim):
            self.label2vec[labels[i]] = y_eyes[i]

        self.loss = []

    def forward(self, x):
        out = np.dot(x, self.weights) + self.bias
        return np.array(list(map(sigmoid, out)))

    def backward(self, x, vec_y, alpha=0.001):
        # print('backward --- ---- ---')
        _y = self.forward(x)

        # diff = _y-vec_y
        # l = 0
        # print(diff)
        # for d in diff:
        #     l += reduce(lambda x,y: np.log(x*x) + np.log(y*y), d)
        # self.loss.append(l)

        # print(self.gratitude_w)
        for i in range(len(y)):
            k = vec_y[i].argmax()
            # print(vec_y[i], k)
            for j in range(self.y_dim):
                # print('i,j = ', i,j)
                if j == k:
                    # print('j=k')
                    self.gratitude_w[j] += (1-_y[i, j])*x[i]
                    self.gratitude_b[j] += (1-_y[i, j])*1
                    # print(1-_y[i,j])
                else:
                    # s =1
                    self.gratitude_w[j] += -1*_y[i, j]*x[i]
                    self.gratitude_b[j] += -1*_y[i, j]*1
                    # print(_y[i,j])
                # print(self.gratitude_w)
                # print(self.gratitude_b)

            # print('-----------------------------------------------------')

        # print(self.gratitude_b)
        self.weights -= alpha * self.gratitude_w
        self.bias -= alpha * self.gratitude_b

        # yt = self.forward(x)
        # print(yt)
        # print(vec_y)

        # print(self.bias)
        self.gratitude_w = np.zeros(self.weights.shape)
        self.gratitude_b = np.zeros(self.bias.shape)

    def fit(self, x, y):
        vec_y = []
        for i in range(len(y)):
            vec_y.append(self.label2vec[y[i]])
        vec_y = np.array(vec_y)
        ret = []
        for _ in range(2500):
            ret.append([self.weights.copy(), self.bias.copy()])
            self.backward(x, vec_y)
        return ret

    def predict(self, x):
        vec = self.forward(x)
        return vec


if __name__ == '__main__':

    raw_input_data = np.loadtxt(
        '../data/' + 'test/y_x_label2.txt', delimiter=',')
    x = raw_input_data[:, :2]
    y = raw_input_data[:, 2]
    label = list(set(y))
    nn = NN_Logistic(len(x[0]), label)

    ret = nn.fit(x, y)

    print(nn.weights)
    print(nn.bias)

    # print(nn.loss)
    print(nn.predict([-0.017612, 14.053064]))

    bestx = np.arange(-3.0, 3.0, 0.1)
    besty = []

    # print(ret)
    for w, b in ret[:]:
        by = -1*(w[0, 0]*bestx + b[0]) / w[0, 1]
        besty.append(by)

    # for theta in theta_list[-1:]:
    #     by = -1*(theta[0]+theta[1]*bestx)/theta[2]
    #     besty.append(by)

    import matplotlib.pyplot as plt

    xcord1, ycord1, xcord2, ycord2 = [], [], [], []
    for i in range(len(x)):
        if y[i] == 1:
            xcord1.append(x[i, 0])
            ycord1.append(x[i, 1])
        else:
            xcord2.append(x[i, 0])
            ycord2.append(x[i, 1])

    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=20, c='red', marker='s', alpha=.5)
    ax.scatter(xcord2, ycord2, s=20, c='green', alpha=.5)

    # ax.plot(bestx,besty)
    for by in besty:
        ax.plot(bestx, by)
        # print(by)

    plt.title('input point')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()
