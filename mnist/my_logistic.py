"""
use the logistic regression written by myself

there are some bugs !!!

"""

import tensorflow as tf 
import numpy as np 
from PIL import Image
import time


mnist = tf.keras.datasets.mnist.load_data()

train_x, train_y = mnist[0][0], mnist[0][1]
test_x, test_y = mnist[1][0], mnist[1][1]

train_linear_x = np.reshape(train_x, [-1,784])
test_linear_x = np.reshape(test_x, [-1,784])


""" use 0/1 to classify every pixel """
train_linear_x = np.sign(train_linear_x)
test_linear_x = np.sign(test_linear_x)

# print(len(train_linear_x), train_linear_x.mean())

time_begin = time.time()

import sys
sys.path.append('../sxlearn')
import logistic
ylabels = list(set(train_y))
lr = logistic.LogisticRegression(train_linear_x.shape[1], ylabels, need_normalized=1)
print(train_linear_x.shape)
lr.feed(train_linear_x, train_y, iteration=1, alpha=0.01)

y_pred = []
print(test_linear_x.shape)

for i in range(test_linear_x.shape[0]):
    print(i)
    y_pred.append(lr.predict([test_linear_x[i]]))


time_end = time.time()

mispredict = (test_y != y_pred).sum()
print("Number of mislabeled points out of a total %d points : %d"
       % (test_linear_x.shape[0],mispredict))

print('accuracy:%f' % (float(test_linear_x.shape[0] - mispredict)/ test_linear_x.shape[0]))
print('train and predict cost %fs' % (time_end-time_begin))