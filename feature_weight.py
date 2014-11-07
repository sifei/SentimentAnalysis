from sklearn.svm import LinearSVC
import data_io,csv
from scipy.sparse import csr_matrix
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import svm,cross_validation
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.cross_validation import cross_val_score
import numpy as np
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC, LinearSVC
from scipy.sparse import csr_matrix
from sklearn.metrics import f1_score,accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
import pandas as pd

tuned_parameters = [{'C':[0.02,0.03,0.04,0.05],'tol':[1e-4,1e-3,1e-2,0.1,0.2,0.3,0.5]}]
n_classes = 3
n_estimators = 30
RANDOM_SEED = 13
def read_train(train_path):
    return pd.read_csv(train_path)

def main(filename,ofilename=None):
    train = read_train(filename)
    train.fillna(0, inplace=True)

    train_sample = train[:].fillna(value = 0)

    feature_names = list(train_sample.columns)
    print len(feature_names)
    feature_names.remove("Sentiment")
    feature_names.remove("Tweet")
    feature_remove = ['JJ_count','NN_count','VB_count','RB_count','JJ_Pcount','NN_Pcount','VB_Pcount','RB_Pcount','JJ_Ncount','NN_Ncount','VB_Ncount','RB_Ncount','JJ_Score','RB_Score','VB_Score','NN_Score','POS_score',
                      'Negation_Count','Positive_words','Negative_words','positive-emo','negative-emo','neutral-emo','capitalized_words','exclamation_words','slang_count','non-english_words_cnt','hashPos','hashNeg','BiScore',
                      'sentiStrengthPos','sentiStrengthNeg']
    for term in feature_remove:
        try:
            feature_names.remove(term)
        except:
            print term
    features = train_sample[feature_names].values
    #features = csr_matrix(features)
    target = train_sample["Sentiment"].values
    X,Y = features, target

    clf = MultinomialNB(alpha=1.0,class_prior=None,fit_prior=True)
    X_MNB = clf.fit(X,Y,sample_weight=1)

    toCSV = []
    for line in X_MNB.feature_log_prob_:
        row = []
        for item in line:
            row.append(item)
        for i in range(len(feature_remove)):
            row.append('1')
        toCSV.append(row)

    with open("apoorv_feature_weight.csv",'wb') as fp:
        a = csv.writer(fp)
        a.writerows(toCSV)

    print "feature weight Done"
