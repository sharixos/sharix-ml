import numpy as np

import sys
sys.path.append('../sxlearn')
import bayes
import text


data_path = '../data/'
if __name__ == '__main__':
    
    documents = []
    f = open(data_path + 'test/feature.txt', 'r')
    y = []
    for line in f:
        strlist = line.replace('\n', '').split(',')
        y.append(strlist[-1])
        document = []
        for i in range(len(strlist[:-1])):
            document.append('x'+str(i) +'_'+strlist[i])
        documents.append(document)
    
    
    wv = text.WordVector()
    for document in documents:
        wv.add_document(document)
    
    train_x = []
    for document in documents:
        train_x.append(wv.onehot_wordvector(document))
    print(train_x)

    nb = bayes.NaiveBayes(len(wv.vocab), y)
    nb.feed(train_x, y)
    nb.feed_end()

    # pred = nb.predict(train_x[1])
    # print(pred, y[1])
    count, count2 = 0, 0
    for i in range(len(train_x)):
        pred = nb.predict(train_x[i])
        if pred == y[i]:
            print('true')
            count += 1
        else:
            print('false')
            count2 += 1
    print(count, count2)
            
