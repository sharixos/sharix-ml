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

    time1 = time.time()

    wv = text.WordVector()

    pathdict = {}


    for path in pathlabels:
        pathdict[path] = []
        for dirpath, dirnames, filenames in os.walk(path):
            for file in filenames:
                fullpath = os.path.join(dirpath, file)
                pathdict[path].append(fullpath)
                
    trainlist,testlist = [],[]
    train_y_list, test_y_list = [],[]
    for path in pathlabels:
        count = len(pathdict[path])
        until = int(count*0.9)
        trainlist.extend(pathdict[path][:until])
        testlist.extend(pathdict[path][until:])
        train_y_list.extend([path]*until)
        test_y_list.extend([path]*(count-until))
        for fullpath in pathdict[path]:
            wv.add_document(wv.get_word_list(fullpath,encoding='gbk'))
        print(len(wv.vocab))



    print('finished establishing words')
    time2 = time.time()
    
    print('getting document vector and feed nb at the same time')

    nb = bayes.NaiveBayes(len(wv.vocab), pathlabels)

    for i in range(len(trainlist)):
        print('feed:' + trainlist[i])
        vec = wv.onehot_wordvector(wv.get_word_list(trainlist[i], encoding='gbk'))
        nb.feed([vec], [train_y_list[i]])

    
    time3 = time.time()
    print('finished train')

    truecount, falsecount = 0,0
    for i in range(len(testlist)):
        vec = wv.onehot_wordvector(wv.get_word_list(testlist[i], encoding='gbk'))
        pred = nb.predict(vec)
        print(pred, path)
        if pred == path:
            truecount += 1
        else:
            falsecount += 1

    print(truecount, falsecount)
    accuracy = float(truecount)/(truecount + falsecount) 
    print('accuracy:' , accuracy)

    time4 = time.time()

    print('------ time cost list -------')
    print('establish wordlist: %f' % (time2-time1))
    print('training cost: %f' % (time3-time2))
    print('test cost:%f' % (time4-time3))