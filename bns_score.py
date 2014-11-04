from string import Template
from nltk import RegexpTokenizer
import re
import nltk
from scipy import stats
import math
import numpy as np
from nltk.corpus.reader.rte import norm
import string as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from numpy.oldnumeric.random_array import normal
from scipy.stats import norm

class textVectorizering(object):
    def __init__(self,sourceData=None,min_df_= 1,stopwords = None,binary = False):
        self.data = sourceData
        self.vectorizer = CountVectorizer(min_df = min_df_,stop_words = stopwords,binary = binary)
        self.inverseMat = None
        self.feature_list = None
        self.mat = None
        self.list_count = None
        self.zipper = None
        self.dictWord = {}
        self.flatten_list = None
        self.list_doc = []
        self.records = len(self.data)
    def calculate_pair(self):
        self.mat = self.vectorizer.fit_transform(self.data)
        self.feature_list = self.vectorizer.get_feature_names()
        self.inverseMat = self.vectorizer.inverse_transform(self.mat)
        for elem in self.inverseMat:
            self.list_doc.append(elem.flatten().tolist())
        self.list_count = np.array(self.mat.sum(axis = 0))[0].tolist()
        print(self.list_count)
        self.zipper = zip(self.feature_list,self.list_count)
        for elem in self.zipper:
            self.dictWord[elem[0]] =elem[1]
        return
    def count_word(self,word):
        tp= 0
        for each in self.list_doc:
            if(each.count(word) != 0):
                tp += 1
        return tp

def returnFinverse(X):
    if(X < 0.00005):
        X =0.00005
    return norm.ppf(norm.cdf(X))


def getPosScore(TP,FN):
    return np.ndim(TP) + np.ndim(FN)

def getNegScore(FP,TN):
    return np.ndim(FP) + np.ndim(TN)
    #def file_writer(path='C:\\TwitterBigData\\default.txt',data_list):
    #   file_to_write = open(path,'w+')
    #  for elem in data_list:
    #     file_to_write.write(elem)
    #file_to_write.close()



def calculate_bns(list_data,vecP,vecN,dictForList):
    for each in list_data:
        tp,fp,fn,tn = 0,0,0,0
        pos,neg = 0,0
        tp += vecP.dictWord.get(each,0)
        fp += vecN.dictWord.get(each,0)
        print('tp calculation: '+ str(tp))
        print('fp calculation: ' + str(fp))
        print('len of records: ' + str(vecP.records))
        print('len of neg records: ' + str(vecN.records))
        fn += math.fabs((vecP.records - tp))
        tn += math.fabs((vecN.records - fp))

        pos = tp + fn
        neg = fp + tn
        tpr = tp/pos
        fpr = fp/neg
        dictForList[each] = math.fabs(returnFinverse(tpr)-returnFinverse(fpr))
    return dictForList
def mainTest(posX,negX):

    vecP = textVectorizering(sourceData = posX,min_df_= 1,binary=True,stopwords ='english')
    vecN = textVectorizering(sourceData = negX,min_df_ = 1,binary =True,stopwords='english')
    vecP.calculate_pair()
    vecN.calculate_pair()
    dictForlist = {}
    total_features = []
    total_features.extend(vecP.feature_list)
    total_features.extend(vecN.feature_list)
    dict_for_feature = calculate_bns(total_features,vecP,vecN,dictForlist)
    file_bns_output = open('C:\\TwitterBigData\\profileOutput\\bns_score.txt','w+')
    for (key,values) in dict_for_feature.iteritems():
        file_bns_output.write(key+" => "+ str(values))
        file_bns_output.write('\n')
    file_bns_output.close()
    return
def mainSource():

    file_to_read_neg = open('C:\\TwitterBigData\\profileOutput\\topLowRanked120Profile.txt')
    file_to_read_pos = open('C:\\TwitterBigData\\profileOutput\\topRanked60Profile.txt')
    try:
        data_pos = file_to_read_pos.readlines()
    finally:
        file_to_read_pos.close()
    try:
        data_neg = file_to_read_neg.readlines()
    finally:
        file_to_read_neg.close()
    list_of_pos_doc = []
    for datum in data_pos:
        list_of_pos_doc.append(datum.strip().decode('utf-8','ignore'))
    list_of_neg_doc = []
    for datum in data_neg:
        list_of_neg_doc.append(datum.strip().decode('utf-8','ignore'))

    print('list of pos doc size: '+ str(len(list_of_pos_doc)))
    print('list of neg doc size: '+ str(len(list_of_neg_doc)))

    mainTest(list_of_pos_doc,list_of_neg_doc)

#########################################################
# start of a main function
mainSource()
