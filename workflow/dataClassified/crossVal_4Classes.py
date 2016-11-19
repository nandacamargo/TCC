"""
Validação cruzada para classificação em quatro classes, 
usando o SVM como classificador.

Cross-validation to classify into four classes, 
using SVM classifier.
"""

import numpy as np
import sys

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn import metrics
from sklearn import cross_validation


categories = ['neg2', 'neutral', 'pos', 'vpos']

print("Loading for categories:")
print(categories)


allLabeled = '../label/allLabeled.txt' 
path = 'train/FourClasses/'

with open(allLabeled, 'r') as al:
    for file_label in al:
        file_label = file_label.rstrip()
        fname = '_'.join(file_label.split('_')[:-1])

        neg1 = path + 'neg2/' + fname + '_vneg.txt'
        neg2 = path + 'neg2/' + fname + '_neg.txt'
        neutral = path + 'neutral/' + fname + '_neutral.txt'
        pos = path + 'pos/' + fname + '_pos.txt'
        very_pos = path + 'vpos/' + fname + '_vpos.txt'


        print("=====================")
        print("Filme " + fname)
        print("=====================")
        
        n1 = []
        n2 = []
        nn = []
        p = []
        vp = []
        
        n1_target = []
        n2_target = []
        nn_target = []
        p_target = []
        vp_target = []

        tg = [-2, -1, 1, 2]


        with open(neg1, 'r') as fn:
            for line in fn:
                n1.append(line)
                n1_target.append('-1')

        with open(neg2, 'r') as fn:
            for line in fn:
                n2.append(line)
                n2_target.append('-1')

        with open(neutral, 'r') as fn:
            for line in fn:
                nn.append(line)
                nn_target.append('0')

        with open(pos, 'r') as fn:
            for line in fn:
                p.append(line)
                p_target.append('1')


        with open(very_pos, 'r') as fp:
            for line in fp:
                vp.append(line)
                vp_target.append('2')
                

        data_train = n1 + n2 + nn + p + vp
        data_target = n1_target + n2_target + nn_target + p_target + vp_target


        vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
        X_train = vectorizer.fit_transform(data_train)

        clf = svm.SVC(kernel='linear', C=.80)

        scores = cross_validation.cross_val_score(clf, X_train, data_target, cv=10)

        print('scores: ')
        print(scores)
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        print("================================\n")
