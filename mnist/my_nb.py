"""
use the naive bayes written by myself
it turns out that the time cost 19s, but the accuracy is good
    the reason is i use dict to store the weights in my naive bayes model
    to use numpy will be better
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
import bayes
ylabels = list(set(train_y))
nb = bayes.NaiveBayes(train_linear_x.shape[1], ylabels)
nb.feed(train_linear_x, train_y)
y_pred = []
for tx in test_linear_x:
    y_pred.append(nb.predict(tx))

time_end = time.time()

mispredict = (test_y != y_pred).sum()
print("Number of mislabeled points out of a total %d points : %d"
       % (test_linear_x.shape[0],mispredict))

print('accuracy:%f' % (float(test_linear_x.shape[0] - mispredict)/ test_linear_x.shape[0]))
print('train and predict cost %fs' % (time_end-time_begin))