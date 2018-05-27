import tensorflow as tf 
import numpy as np 
from PIL import Image

mnist = tf.keras.datasets.mnist.load_data()

train_x, train_y = mnist[0][0].astype('float32')/255, mnist[0][1]
test_x, test_y = mnist[1][0].astype('float32')/255, mnist[1][1]


onehot_train_y = []
for ty in train_y:
    onehot = [0]*10
    onehot[ty] = 1
    onehot_train_y.append(onehot)

onehot_test_y = []
for ty in test_y:
    onehot = [0]*10
    onehot[ty] = 1
    onehot_test_y.append(onehot)


# img = Image.fromarray(train_x[0])
# img.show()

x_image = tf.placeholder("float", [None, 28,28])

x = tf.reshape(x_image, [-1,784])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W)+b)

y_ = tf.placeholder("float", [None, 10])

cross_entropy = -1 * tf.reduce_sum(y_*tf.log(y))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

iteration = 1
batch_size = 100
num_batchs = len(train_x) / 100
loss_list = []

for i in range(iteration):
    for j in range(num_batchs):
        batch_xs, batch_ys = train_x[100*j:100*(j+1)], onehot_train_y[100*j:100*(j+1)]
        sess.run(train_step, feed_dict={x_image: batch_xs, y_: batch_ys})
        loss_list.append(sess.run(cross_entropy, feed_dict={x_image: batch_xs, y_: batch_ys}))

W = sess.run(W)*255
W = W.transpose()
W2 = np.reshape(np.array(W[0]), [28,28])
for w in W[1:]:
    W2 = np.concatenate((W2, np.reshape(np.array(w.copy()), [28,28])), axis=1)
img = Image.fromarray(W2)
img.show()

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
# print(loss_list)

import matplotlib.pyplot as plt
ax = np.linspace(1, iteration * num_batchs, iteration * num_batchs)
plt.figure(1)
plt.plot(ax, loss_list)
plt.show()

print sess.run(accuracy, feed_dict={x_image: test_x, y_: onehot_test_y})
