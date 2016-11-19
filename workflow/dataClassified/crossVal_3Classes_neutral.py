"""
Validação cruzada para classificação em positivo, negativo 
ou neutro, usando o SVM como classificador.

Cross-validation to classify in positive, negative or neutral  
using SVM classifier.
"""

import numpy as np
import sys
from time import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.utils.extmath import density
from sklearn import metrics
from sklearn import cross_validation
import random


categories = ['neg', 'pos', 'noise']

print("Loading for categories:")
print(categories)


allLabeled = '../label/allLabeled.txt' 

path = 'train/ThreeClasses/'
path2 = 'train/FiveClasses/'

with open(allLabeled, 'r') as al:
    for file_label in al:
        file_label = file_label.rstrip()
        fname = '_'.join(file_label.split('_')[:-1])
        
        negfiles = path + 'neg/' + fname + '_neg.txt'
        posfiles = path + 'pos/' + fname + '_pos.txt'
        noisefiles = path2 + 'neutral/' + fname + '_neutral.txt'


        print("=====================")
        print("Filme " + fname)
        print("=====================")
        
        n = []
        p = []
        ns = []
        n_target = []
        p_target = []
        ns_target = []

        tg = [-1, 1, 5]

        with open(negfiles, 'r') as fn:
            for line in fn:
                n.append(line)
                n_target.append('-1')

        with open(noisefiles, 'r') as fns:
            for line in fns:
                ns.append(line)
                ns_target.append('5')

        with open(posfiles, 'r') as fp:
            for line in fp:
                p.append(line)
                p_target.append('1')


        data_train = n + p + ns 
        data_target = n_target + p_target + ns_target
 
        vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
        X_train = vectorizer.fit_transform(data_train)

        clf = svm.SVC(kernel='linear', C=1)
        scores = cross_validation.cross_val_score(clf, X_train, data_target, cv=10)

        print('scores: ')
        print(scores)
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        print("================================\n")

