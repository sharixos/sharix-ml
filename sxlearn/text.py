import os,jieba,re
import numpy as np

class WordVector(object):
    """
    
    Author: [@Wang Xin Gang](https://github.com/sharixos)

    Website: http://www.sharix.site/
    
    Github: https://github.com/sharixos

    Attributes:
        self.vocab: the whole vacabulary
    """
    def __init__(self):
        self.vocab = set([])

    def add_document(self,document):
        """
        Params:
            document: a list of word
        """
        self.vocab |= set(document)

    def get_word_list(self,fullpath,encoding='utf8'):
        f = open(fullpath, 'r',encoding=encoding,errors='ignore')
        content = re.sub('\W',' ',f.read()).lower()
        wordlist = jieba.lcut(content, cut_all=False)
        while ' ' in wordlist:
            wordlist.remove(' ')
        f.close()
        return wordlist

    def onehot_wordvector(self,document):
        vocab = list(self.vocab)
        vec = [0] * len(vocab)
        for word in document:
            if word in vocab:
                vec[vocab.index(word)] = 1
            else:
                print('error: the word not found % ' % word)
        return vec