import tensorflow as tf
from sklearn import datasets
import numpy as np 

boston = datasets.load_boston()
x = boston.data.astype('float32')
y = boston.target.astype('float32')
m,n = np.shape(x)


for i in range(n):
    minxi = np.min(x[:,i])
    maxxi = np.max(x[:,i])
    x[:,i] = (x[:,i] - minxi)  / (maxxi - minxi)

train_x = tf.placeholder(tf.float32, [m,n], name='train_x')
train_y = tf.placeholder(tf.float32, [m,], name='train_y')

b = tf.Variable(tf.zeros([1]))
W = tf.Variable(tf.random_uniform([1,n], -1.0, 1.0))
_y = tf.matmul(train_x,W, transpose_b=True) + b

loss = tf.reduce_mean(tf.square(_y - train_y))
optimizer = tf.train.GradientDescentOptimizer(0.1)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

loss_list = []
with tf.Session() as sess:
    sess.run(init)
    for step in range(200):
        sess.run(train, feed_dict={train_x:x, train_y:y})
        loss_list.append(sess.run(loss, feed_dict={train_x:x, train_y:y}))

import matplotlib.pyplot as plt
ax = np.linspace(1, 200, 200)
plt.figure(1)
plt.plot(ax, loss_list)
plt.show()
