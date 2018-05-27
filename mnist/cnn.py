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




def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

x = tf.placeholder("float", [None,28,28]) 
y_ = tf.placeholder("float", [None, 10])

W_conv1 = weight_variable([5,5,1,8])
b_conv1 = bias_variable([8])
x_image = tf.reshape(x, [-1,28,28,1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# layer 2
W_conv2 = weight_variable([5,5,8,16])
b_conv2 = bias_variable([16])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)


# layer 3
W_fc1 = weight_variable([7*7*16, 128])
b_fc1 = bias_variable([128])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*16])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
print(h_fc1_drop)
W_fc2 = weight_variable([128, 10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

print(y_conv)

cross_entropy = -1 * tf.reduce_sum(y_*tf.log(y_conv))

train_step = tf.train.GradientDescentOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


iteration = 10
batch_size = 50

# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     for i in range(iteration):
#         for j in range(100):
#             # print(i,j)
#             batch_xs, batch_ys = train_x[batch_size*j:batch_size*(j+1)], onehot_train_y[batch_size*j:batch_size*(j+1)]
#             train_step.run(feed_dict={x: batch_xs, y_:batch_ys, keep_prob:0.5})

#         train_accuracy = accuracy.eval(feed_dict={x: batch_xs, y_:batch_ys, keep_prob:1.0})
#         print("step %d, training accuracy %g"%(i, train_accuracy))
#     print("test accuracy %g"%accuracy.eval(feed_dict={x: test_x, y_: onehot_test_y, keep_prob: 1.0}))
