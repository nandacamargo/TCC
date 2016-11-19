"""
Classification in positive or negative the data from movies with
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
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.utils.extmath import density
from sklearn import metrics
from nltk.corpus import movie_reviews
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

    fname = ('_').join(f.split('_')[:-1]) 
    print("fname {}".format(fname))
    classifier = string + '/'
    pmovie = '../results/pos/' + classifier + fname + '_pos.json'
    nmovie = '../results/neg/' + classifier + fname + '_neg.json'

    real_names = {"Alice": "Alice Through the Looking Glass", "Deadpool": "Deadpool", "Finding_Dory": "Finding Dory", "Huntsman": "The Huntsman: Winter's War", "Mogli": "The Jungle Book", "Ratchet": "Ratchet & Clank", "Revenant": "The Revenant", "Warcraft": "Warcraft: The Beginning", "War": "Captain America: Civil War", "Zootopia": "Zootopia"}

    with open(pmovie, 'w') as p, open(nmovie, 'w') as n:
        i = 0
        for doc, category in zip(data_test, pred):
            #print('%r => %s' % (doc, categories[category]))
            newline = ',\n\t"movie": "' + real_names[fname] + '",\n\t"label": "' + category + '"\n    },\n    "type": "Feature"\n},\n'
            block_list[i] = block_list[i].rstrip() + newline
            if (category == 'pos'):
                p.write(block_list[i])
            elif (category == 'neg'):
                n.write(block_list[i])
            i+=1

    
    if hasattr(clf, 'coef_'):
        print("dimensionality: %d" % clf.coef_.shape[1])
        print("density: %f" % density(clf.coef_))

    
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
def size_mb(docs):
    return sum(len(s.encode('utf-8')) for s in docs) / 1e6
###############################################################################

# Load some categories from the training set
categories = ['pos', 'neg']

print("Loading for categories:")
print(categories if categories else "all")

path = 'train/TwoClasses/'

ntfile = path + 'neg/all_neg.txt'
ptfile = path + 'pos/all_pos.txt'

print("==================================")
print("Classificando filmes ")
print("Isso pode demorar um pouco...")
print("==================================")

n = [] 
p = []
n_target = []
p_target = []

with open(ntfile, 'r') as nf:
    for line in nf:
        n.append(line)
        n_target.append('neg')


with open(ptfile, 'r') as pf:
    for line in pf:
        p.append(line)
        p_target.append('pos')



data_train = n + p
y_train = n_target + p_target

categories = sorted(categories)


data_test = [] 
test_list = ['Alice_notNoise.json', 'Deadpool_notNoise.json', 'Finding_Dory_notNoise.json', 'Huntsman_notNoise.json', 'Mogli_notNoise.json', 'Ratchet_notNoise.json', 'Revenant_notNoise.json', 'Warcraft_notNoise.json', 'War_notNoise.json', 'Zootopia_notNoise.json']
path2 = 'test/notNoise/'

classifiers = ["SVM", "NB"]

for f in test_list:
    for clf in classifiers:

        classifier = clf + '/'
        ftest = path2 + classifier + f

        print("ftest = {}".format(ftest))
        
        block_list = []
        block = ""
        data_test = []

        with open(ftest, 'r') as ft:

            print("==================================")
            print("Processando file: {}".format(ftest))
            print("==================================")

            cont = 0
            for line in ft:
                if ('"tweet":' in line): 
                    #get the part of line after :
                    t = line.split(':')[1]
                    data_test.append(t)
               
                if ('}' in line):
                    cont+= 1
      
               
                if ('}' in line and cont == 2):
                    cont = 0
                    block_list.append(block)
                    block = ""
                    tmp = ft.readline()
                    tmp = ft.readline()
                    tmp = ft.readline()

                else:
                    block += line


                if ('properties' in line):
                    newline =  '\t"classifier": "' + clf + '",\n'
                    block += newline

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

        def trim(s):
            """Trim string to fit on terminal (assuming 80-column display)"""
            return s if len(s) <= 80 else s[:77] + "..."
       
         # mapping from integer feature name to original token string
        feature_names = vectorizer.get_feature_names()
        feature_names = np.asarray(feature_names)

        if ("SVM" == clf):
            # Train sparse SVM classifiers
            print('=' * 80)
            print("SVM Linear")
            classification_metrics(LinearSVC(C=0.7, multi_class='ovr'), "SVM")

        else:
            # Train sparse Naive Bayes classifiers
            print('=' * 80)
            print("Naive Bayes")
            classification_metrics(MultinomialNB(alpha=0.1), "NB")
