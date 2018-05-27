import numpy as np
from functools import reduce


class NaiveBayes(object):
    """

    Author: [@Wang Xin Gang](https://github.com/sharixos)

    Website: http://www.sharix.site/

    Github: https://github.com/sharixos

    Args:
        num_x: the dimension of vector
        labels: the list of label
    Attributes:
        num_y_dict: statistical information of y
        sum_xvec_dict: sum of x vector
        hot_sum_dict: sum of total hots

        py_dict : porportion of y
        regularized_xvec_dict: sum_xvec_dict/hot_sum_dict
    """

    def __init__(self, num_x, labels):
        self.num_x = num_x
        self.num_y = len(labels)
        self.labels = labels
        self.num_training = 0

        self.num_y_dict = {}
        self.sum_xvec_dict = {}
        self.hot_sum_dict = {}

        for l in self.labels:
            self.num_y_dict[l] = 0
            self.sum_xvec_dict[l] = np.ones(num_x)  # Laplace smooth +1
            self.hot_sum_dict[l] = num_x

        self.py_dict = {}
        self.regularized_xvec_dict = {}

    def feed(self, train_x, train_y):
        self.num_training += len(train_x)
        for i in range(len(train_x)):
            self.num_y_dict[train_y[i]] += 1
            self.sum_xvec_dict[train_y[i]] += train_x[i]
            self.hot_sum_dict[train_y[i]] += sum(train_x[i])

        # update parameter
        for l in self.labels:
            self.py_dict[l] = np.log(
                float(self.num_y_dict[l]) / self.num_training)
            self.regularized_xvec_dict[l] = np.log(
                self.sum_xvec_dict[l] / self.hot_sum_dict[l])

    def predict(self, inputx):
        """

        """
        x = np.array(inputx)
        result = {}
        for l in self.labels:
            px = sum(self.regularized_xvec_dict[l] * x)
            result[l] = self.py_dict[l] + px  # use log so here is +
        return sorted(result, key=lambda x: result[x])[-1]  # return max
