import tensorflow as tf 
import numpy as np 
from PIL import Image
import time


mnist = tf.keras.datasets.mnist.load_data()

train_x, train_y = mnist[0][0], mnist[0][1]
test_x, test_y = mnist[1][0], mnist[1][1]

train_linear_x = np.reshape(train_x, [-1,784])
test_linear_x = np.reshape(test_x, [-1,784])



from sklearn import datasets
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB


time_begin = time.time()


""" use Gaussian Naive Bayes
    accuracy: 0.55, time 1.9s    bad """
# gnb = GaussianNB()
# gnb.fit(train_linear_x, train_y)
# y_pred = gnb.predict(test_linear_x)

""" use MultinomialNB NB
    accuracy: 0.83 time 0.4s    nice! """
# mbn = MultinomialNB()
# mbn.fit(train_linear_x, train_y)
# y_pred = mbn.predict(test_linear_x)

""" use Bernoulli Naive Bayes
    accuracy: 0.84 time 0.7s    nice! """
bnb = BernoulliNB()
bnb.fit(train_linear_x, train_y)
y_pred = bnb.predict(test_linear_x)

time_end = time.time()

mispredict = (test_y != y_pred).sum()
print("Number of mislabeled points out of a total %d points : %d"
       % (test_linear_x.shape[0],mispredict))

print('accuracy:%f' % (float(test_linear_x.shape[0] - mispredict)/ test_linear_x.shape[0]))
print('train and predict cost %fs' % (time_end-time_begin))