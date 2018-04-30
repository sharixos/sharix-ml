"""
    Author: [@Wang Xin Gang](https://github.com/sharixos)

    Website: http://www.sharix.site/
    
    Github: https://github.com/sharixos
"""

import os
import jieba
import re
import numpy as np

import sys
sys.path.append('../sxlearn')
import logistic
import text

# data8 = '../data/sogou/C000008/'
# data10 = '../data/sogou/C000010/'
# data13 = '../data/sogou/C000013/'
# data14 = '../data/sogou/C000014/'
# data16 = '../data/sogou/C000016/'
# data20 = '../data/sogou/C000020/'
# data22 = '../data/sogou/C000022/'
# data23 = '../data/sogou/C000023/'
# data24 = '../data/sogou/C000024/'

data1 = '../data/sogoumini/1/'
data2 = '../data/sogoumini/2/'
data3 = '../data/sogoumini/3/'
data4 = '../data/sogoumini/4/'

trainpercent = 0.9
if __name__ == '__main__':
    # pathlabels = [data8, data10, data13, data14,
    #               data16, data20, data22, data23, data24]
    pathlabels = [data1, data2, data3,data4]


    wv = text.WordVector()

    pathdict = {}

    for path in pathlabels:
        pathdict[path] = []
        for dirpath, dirnames, filenames in os.walk(path):
            for file in filenames:
                fullpath = os.path.join(dirpath, file)
                pathdict[path].append(fullpath)
                
    
    for path in pathlabels:
        for fullpath in pathdict[path]:
            print(fullpath, len(wv.vocab))
            wv.add_document(wv.get_word_list(fullpath,encoding='gbk'))

    print('finished establishing words')

    print('getting document vector')
    trainset_dict, testset_dict = {}, {}
    vectordict = {}
    for path in pathlabels:
        vectordict[path] = []
        print(path)
        for fullpath in pathdict[path]:
            print('get vector:', fullpath)
            vec = wv.onehot_wordvector(wv.get_word_list(fullpath, encoding='gbk'))
            vectordict[path].append(vec)
        until = int(len(vectordict[path])*0.9)
        print(until)
        trainset_dict[path] = vectordict[path][:until]
        testset_dict[path] = vectordict[path][until:]
    
    print('finished get train and test set')

    lr = logistic.LogisticRegression(len(wv.vocab), pathlabels)

    # prepare train_x and train_y, the data need to be shuffled
    train = []
    for path in pathlabels:
        for x in trainset_dict[path]:
            train.append([x,path])
    
    import random
    random.shuffle(train)

    train_x, train_y = [],[]
    for x,y in train:
        train_x.append(x)
        train_y.append(y)
    cost_list,theta_list = lr.feed(train_x,train_y,iteration=50,alpha=0.001)
    print(theta_list)
    print(cost_list)

    truecount, falsecount = 0,0
    for path in pathlabels:
        for vec in testset_dict[path]:
            pred = lr.predict([vec])
            print(pred, path)
            if pred == path:
                truecount += 1
            else:
                falsecount += 1

    print(truecount, falsecount)
    accuracy = float(truecount)/(truecount + falsecount) 
    print('accuracy:' , accuracy)

    import matplotlib.pyplot as plt

    ax = np.linspace(1, 50, 50)
    plt.figure(1)
    plt.plot(ax, cost_list)
    plt.show()