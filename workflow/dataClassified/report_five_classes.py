"""
Utilizado para calcular precisão, cobertura e medida f
das classes muito positivo, positivo, neutro, negativo 
e muito negativo com classificador SVM.

Calculate precision, recall and f-measure to classes 
very positive, positive, neutral, negative and very
negative with SVM classifier.

"""

#Based on example:
#http://scikit-learn.org/stable/auto_examples/text/document_classification_20newsgroups.html#example-text-document-classification-20newsgroups-py

from __future__ import print_function

import logging
import numpy as np
from optparse import OptionParser
import sys
from time import time


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC as svm
from sklearn import svm
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.utils.extmath import density
from sklearn import metrics
from sklearn import cross_validation
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn import preprocessing
from sklearn import preprocessing
import random

from scipy.sparse import coo_matrix

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


op = OptionParser()
op.add_option("--report",
              action="store_true", dest="print_report",
              help="Print a detailed classification report.")
op.add_option("--confusion_matrix",
              action="store_true", dest="print_cm",
              help="Print the confusion matrix.")


(opts, args) = op.parse_args()
if len(args) > 0:
    op.error("This script takes no arguments.")
    sys.exit(1)

print(__doc__)
op.print_help()
print()


categories = ['vneg', 'neg', 'neutral', 'pos', 'vpos']


print("Loading for categories:")
print(categories)


allLabeled = '../label/allLabeled.txt' 
#allLabeled = 'todos.txt'
path = 'train/FiveClasses/'

with open(allLabeled, 'r') as al:
    for file_label in al:
        file_label = file_label.rstrip()
        fname = '_'.join(file_label.split('_')[:-1])
        vnegfiles = path + 'vneg/' + fname + '_vneg.txt'
        negfiles = path + 'neg/' + fname + '_neg.txt'
        neutralfiles = path + 'neutral/' + fname + '_neutral.txt'
        posfiles = path + 'pos/' + fname + '_pos.txt'
        vposfiles = path + 'vpos/' + fname + '_vpos.txt'

        print("=====================")
        print("Filme " + fname)
        print("=====================")
        
        vn = []
        n = []
        neutral = []
        p = []
        vp = []        

        vn_target = []
        n_target = []
        neutral_target = []
        p_target = []
        vp_target = []

        tg = [-2, -1, 0, 1, 2]


        with open(vnegfiles, 'r') as fvn:
           for line in fvn:
                vn.append(line)
                vn_target.append('-2')

        with open(negfiles, 'r') as fn:
           for line in fn:
                n.append(line)
                n_target.append('-1')

        with open(neutralfiles, 'r') as fnn:
            for line in fnn:
                neutral.append(line)
                neutral_target.append('0')

        with open(posfiles, 'r') as fp:
            for line in fp:
                p.append(line)
                p_target.append('1')

        with open(vposfiles, 'r') as fvp:
            for line in fvp:
                vp.append(line)
                vp_target.append('2')


        data = np.array(vn + n + neutral + p + vp)
        data_target = np.array(vn_target + n_target + neutral_target + p_target + vp_target)

        #skf = StratifiedKFold(data_target, n_folds=2, shuffle=True)
        sss = StratifiedShuffleSplit(data_target, 10, test_size=0.4, random_state=0)

        data_train = []
        data_test = []
      
        for train_index, test_index in sss:
            data_train, data_test = data[train_index], data[test_index]
            y_train, y_test = data_target[train_index], data_target[test_index]
    
        #print("Tamanho do train: {}".format(len(data_train)))
        #print("Tamanho do teste: {}".format(len(data_test)))

        t0 = time()
        vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')

        X_train = vectorizer.fit_transform(data_train)
        X_test = vectorizer.transform(data_test)


        def classification_metrics(clf):
            print("Training: ")
            print(clf)
            t0 = time()
            clf.fit(X_train, y_train)
            train_time = time() - t0
            print("train time: %0.3fs" % train_time)

            t0 = time()
            pred = clf.predict(X_test)
            test_time = time() - t0
            print("test time:  %0.3fs" % test_time)

            if opts.print_report:
                print("classification report:")
                print(metrics.classification_report(y_test, pred,
                                                target_names=categories))

            if opts.print_cm:
                print("confusion matrix:")
                print(metrics.confusion_matrix(y_test, pred))

            scores = cross_validation.cross_val_score(clf, X_train, y_train, cv=10)
            
            print('scores: ')
            print(scores)
            print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
            print("================================\n")


        # SVM
        print('=' * 80)
        print("SVM Linear")
        classification_metrics(svm.LinearSVC(C=0.7, multi_class='ovr'))

        # Naive Bayes
        print('=' * 80)
        print("Naive Bayes")
        classification_metrics(MultinomialNB(alpha=0.1))

