import tensorflow as tf 
import numpy as np 
from PIL import Image
import time


mnist = tf.keras.datasets.mnist.load_data()

train_x, train_y = mnist[0][0], mnist[0][1]
test_x, test_y = mnist[1][0], mnist[1][1]

train_linear_x = np.reshape(train_x, [-1,784])
test_linear_x = np.reshape(test_x, [-1,784])

from sklearn import linear_model

time_begin = time.time()

""" use LogisticRegression
    iter=20     train time = 220.492734 logistic score 0.918800
    iter=10     train time = 38.447340  logistic score 0.917500
    iter=5      train time = 13.598632  logistic score 0.903200
    iter=4      train time = 10.745325  logistic score 0.886800
    iter=3      train time = 8.031269   logistic score 0.852800
"""
# logistic = linear_model.LogisticRegression(max_iter=4)
# logistic.fit(train_linear_x, train_y)

""" use LogisticRegressionCV
    iter=20     train time = 220.492734 logistic score 0.918800
    iter=10     train time = 38.447340  logistic score 0.917500
    iter=5      train time = 13.598632  logistic score 0.903200
    iter=4      train time = 10.745325  logistic score 0.886800
    iter=3      train time = 8.031269   logistic score 0.852800
"""
logistic = linear_model.LogisticRegression(max_iter=1)
iteration = 5
batch_size = 100
num_batchs = len(train_x) / 100

for i in range(iteration):
    for j in range(num_batchs):
        batch_xs, batch_ys = train_linear_x[batch_size*j:batch_size*(j+1)], train_y[batch_size*j:batch_size*(j+1)]
        logistic.fit(batch_xs, batch_ys)


time_end = time.time()

print('train time = %f' % (time_end-time_begin))
print('logistic score %f' % logistic.score(test_linear_x, test_y))