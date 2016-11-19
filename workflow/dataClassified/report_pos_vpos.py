"""
Utilizado para calcular precisão, cobertura e medida f
das classes positivo e muito positivo com classificador
SVM com a opção --report.

Calculate precision, recall and f-measure to classes 
positive and very positive with SVM classifier, using option
--report.

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
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.utils.extmath import density
from sklearn import metrics
from sklearn import cross_validation
from sklearn import preprocessing
import random

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


# parse commandline arguments
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


###############################################################################
categories = ['pos', 'vpos']


print("Loading for categories:")
print(categories)


allLabeled = '../label/allLabeled.txt' 
path = 'train/FiveClasses/'

with open(allLabeled, 'r') as al:
    for file_label in al:
        file_label = file_label.rstrip()
        fname = '_'.join(file_label.split('_')[:-1]) 
        posfiles = path + 'pos/' + fname + '_pos.txt'
        vposfiles = path + 'vpos/' + fname + '_vpos.txt'

        print("=====================")
        print("Filme " + fname)
        print("=====================")
        
        p = []
        vp = []
        
        p_target = []
        vp_target = []

        tg = [1, 2]


        with open(posfiles, 'r') as fp:
            for line in fp:
                p.append(line)
                p_target.append('1')

        with open(vposfiles, 'r') as fvp:
            for line in fvp:
                vp.append(line)
                vp_target.append('2')


        poscutoff = (int) (len(p)*3/4)
        vposcutoff = (int) (len(vp)*3/4)

        data_train = p[:poscutoff] + vp[:vposcutoff]
        y_train = p_target[:poscutoff] + vp_target[:vposcutoff]
        
        data_test = p[poscutoff:] + vp[vposcutoff:]
        y_test = p_target[poscutoff:] +  vp_target[vposcutoff:]

        c = list(zip(data_train, y_train))
        random.shuffle(c)
        data_train, y_train = zip(*c)

        c = list(zip(data_test, y_test))
        random.shuffle(c)
        data_test, y_test = zip(*c)

        t0 = time()
        vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')

        X_train = vectorizer.fit_transform(data_train)
        X_test = vectorizer.transform(data_test)

        clf = svm.LinearSVC(C=0.7, multi_class='ovr')

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


        scores = cross_validation.cross_val_score(clf, X_train, y_train, cv=10)
        
        print('scores: ')
        print(scores)
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        print("================================\n")


        if opts.print_report:
            print("classification report:")
            print(metrics.classification_report(y_test, pred,
                                            target_names=categories))

        if opts.print_cm:
            print("confusion matrix:")
            print(metrics.confusion_matrix(y_test, pred))


