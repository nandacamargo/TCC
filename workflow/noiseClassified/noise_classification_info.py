"""
Classification in noise or not noise the data from movies with
information about localization
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
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.utils.extmath import density
from sklearn import metrics
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


print(__doc__)
op.print_help()
print()



###############################################################################
# Benchmark classifiers
def classification_metrics(clf, string):
    print('_' * 80)
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


    fname = ftest.split('.')[0]
    fname = fname.split('/')[-1]
    fname = ('_').join(fname.split('_')[1:])

    print(fname)
    classifier = string + '/'
    nmovie = 'noise/info/' + classifier + fname + '_noise.json'
    nnmovie = 'notNoise/info/' + classifier + fname + '_notNoise.json'
    print("nmovie ={}".format(nmovie))

    with open(nmovie, 'w') as n, open(nnmovie, 'w') as nn:
        i = 0
        for doc, category in zip(data_test, pred):
            if (category == 'noise'):
                n.write(block_list[i])
            else:
                nn.write(block_list[i])
            i+=1


    if opts.print_report:
        print("classification report:")
        print(metrics.classification_report(y_test, pred,
                                            target_names=categories))

    if opts.print_cm:
        print("confusion matrix:")
        print(metrics.confusion_matrix(y_test, pred))

    print()
    clf_descr = str(clf).split('(')[0]
    return clf_descr, train_time, test_time


###############################################################################
# Load some categories from the training set
categories = ['noise', 'notNoise']

print("Loading for categories:")
print(categories if categories else "all")

path = '../label/'
path2 = '../nLabel/info/'

ntfile = path + 'noise_train/all_noise.txt'
nntfile = path + 'notNoise_train/all_notNoise.txt'


n = [] 
nn = []
n_target = []
nn_target = []

with open(ntfile, 'r') as nf:
    for line in nf:
        n.append(line)
        n_target.append('noise')


with open(nntfile, 'r') as pf:
    for line in pf:
        nn.append(line)
        nn_target.append('notNoise')


data_train = n + nn
y_train = n_target + nn_target

categories = sorted(categories)


def size_mb(docs):
    return sum(len(s.encode('utf-8')) for s in docs) / 1e6


test_list = ['stream_Alice.json', 'stream_Deadpool.json', 'stream_Finding_Dory.json', 'stream_Huntsman.json', 'stream_Mogli.json', 'stream_Ratchet.json', 'stream_Revenant.json', 'stream_Warcraft.json', 'stream_War.json', 'stream_Zootopia.json']

for f in test_list:
    ftest = path2 + f
    block_list = []
    block = ""
    data_test = []

    with open(ftest, 'r') as ft:

        print("==================================")
        print("Processando file: {}".format(ftest))
        print("==================================")

        cont = 0
        for line in ft:
            cont+= 1
            if (cont == 3): break

        cont = 0
        for line in ft:
            if ('"tweet":' in line): 
                t = line.split(':')[1]
                data_test.append(t)
            block += line

            if('}' in line):
                cont+=1

            if ('}' in line and cont == 2):
                cont = 0
                block_list.append(block)
                block = ""


    data_train_size_mb = size_mb(data_train)
    data_test_size_mb = size_mb(data_test)


    print("%d documents - %0.3fMB (training set)" % (
        len(data_train), data_train_size_mb))
    print("%d documents - %0.3fMB (test set)" % (
        len(data_test), data_test_size_mb))

    print("%d categories" % len(categories))
    print()

    print("Extracting features from the training data using a sparse vectorizer")
    t0 = time()
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
    X_train = vectorizer.fit_transform(data_train)

    duration = time() - t0
    print("done in %fs at %0.3fMB/s" % (duration, data_train_size_mb / duration))
    print("n_samples: %d, n_features: %d" % X_train.shape)
    print()

    print("Extracting features from the test data using the same vectorizer")
    t0 = time()

    X_test = vectorizer.transform(data_test)

    duration = time() - t0
    print("done in %fs at %0.3fMB/s" % (duration, data_test_size_mb / duration))
    print("n_samples: %d, n_features: %d" % X_test.shape)
    print()


    # Train sparse SVM classifiers
    print('=' * 80)
    print("SVM Linear")
    classification_metrics(LinearSVC(C=0.7, multi_class='ovr'), "SVM")

    # Train sparse Naive Bayes classifiers
    print('=' * 80)
    print("Naive Bayes")
    classification_metrics(MultinomialNB(alpha=0.1), "NB")

