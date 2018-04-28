import os
import jieba
import re
import numpy as np

import sys
sys.path.append('../sxlearn')
import bayes
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
    pathlabels = [data1, data2, data3, data4]


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

    nb = bayes.NaiveBayes(len(wv.vocab), pathlabels)
    for path in pathlabels:
        nb.feed(trainset_dict[path], [path] * len(trainset_dict[path]))
    nb.feed_end()

    truecount, falsecount = 0,0
    for path in pathlabels:
        for vec in testset_dict[path]:
            pred = nb.predict(vec)
            print(pred, path)
            if pred == path:
                truecount += 1
            else:
                falsecount += 1

    print(truecount, falsecount)
    accuracy = float(truecount)/(truecount + falsecount) 
    print('accuracy:' , accuracy)
