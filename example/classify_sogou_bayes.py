"""
    Author: Wang Xin Gang

    Website: http://www.sharix.site/
    
    Github: https://github.com/sharixos
"""


import os
import jieba
import re
import numpy as np

import sys,time
sys.path.append('../sxlearn')
import bayes
import text


data1 = '../data/sogoumini/1/'
data2 = '../data/sogoumini/2/'
data3 = '../data/sogoumini/3/'
data4 = '../data/sogoumini/4/'

trainpercent = 0.9
if __name__ == '__main__':
    pathlabels = [data1, data2, data3, data4]

    time1 = time.time()

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
    time2 = time.time()

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
    
    time3 = time.time()
    print('finished get train and test set')

    nb = bayes.NaiveBayes(len(wv.vocab), pathlabels)
    for path in pathlabels:
        nb.feed(trainset_dict[path], [path] * len(trainset_dict[path]))

    time4 = time.time()
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

    time5 = time.time()

    print('------ time cost list -------')
    print('establish wordlist: %f' % (time2-time1))
    print('get train and test one-hot vector: %f' % (time3-time2))
    print('training cost: %f' % (time4-time3))
    print('test cost:%f' % (time5-time4))