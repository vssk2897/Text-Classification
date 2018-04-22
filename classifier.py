# Author: V S S Karthik <vssk2897@gmail.com>
# License: MIT Open Source

from __future__ import print_function

import numpy as np
from optparse import OptionParser
import sys
from time import time
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics




# parse commandline arguments
op = OptionParser()
# Automatically adding the Command Line Arguments
op.add_option("--report",
              action="store_true", dest="print_report",
              help="Print a detailed classification report.")
op.add_option("--confusion_matrix",
              action="store_true", dest="print_cm",
              help="Print the confusion matrix.")
op.add_option("--top10",
              action="store_true", dest="print_top10",
              help="Print ten most discriminative terms per class"
                   " for every classifier.")
op.add_option("--all_categories",
              action="store_true", dest="all_categories",
              help="Whether to use all categories or not.")
op.add_option("--use_hashing",
              action="store_true",
              help="Use a hashing vectorizer.")
op.add_option("--n_features",
              action="store", type=int, default=2 ** 16,
              help="n_features when using the hashing vectorizer.")
op.add_option("--filtered",
              action="store_true",
              help="Remove newsgroup information that is easily overfit: "
                   "headers, signatures, and quoting.")


def is_interactive():
    return not hasattr(sys.modules['__main__'], '__file__')

# work-around for Jupyter notebook and IPython console
argv = [] if is_interactive() else sys.argv[1:]
(opts, args) = op.parse_args(argv)
if len(args) > 0:
    op.error("This script takes no arguments.")
    sys.exit(1)

print(__doc__)
op.print_help()


# The Real Fun starts here


#  Load some categories from the training set
if opts.all_categories:
    categories = None
# Working on 4 Class Lables
else:
    categories = [
        'alt.atheism',
        'talk.religion.misc',
        'comp.graphics',
        'sci.space',
    ]
# Removing Headers , footers , quotes
if opts.filtered:
    remove = ('headers', 'footers', 'quotes')
else:
    remove = ()

print("Loading 20 newsgroups dataset for categories:")
print(categories if categories else "all")

data_train = fetch_20newsgroups(subset='train', categories=categories,
                                shuffle=True, random_state=42,
                                remove=remove)

data_test = fetch_20newsgroups(subset='test', categories=categories,
                               shuffle=True, random_state=42,
                               remove=remove)
print('data loaded')


# order of labels in `target_names` can be different from `categories`
target_names = data_train.target_names

# inline function for Calculating Size of file
def size_mb(docs):
    return sum(len(s.encode('utf-8')) for s in docs) / 1e6

data_train_size_mb = size_mb(data_train.data)
data_test_size_mb = size_mb(data_test.data)

# Printing the size of data being operated .
print("{0} documents - {1}MB (training set)".format(len(data_train.data), data_train_size_mb))
print("{0} documents - {1}MB (test set)".format(len(data_test.data), data_test_size_mb))
print(categories)



# split a training set and a test set
y_train, y_test = data_train.target, data_test.target

print("Extracting features from the training data using a sparse vectorizer")
t0 = time()
# Default Parameter passed is to use hash based techniques for reducing the training time
# by not operaing on sparse vectors
if opts.use_hashing:
    vectorizer = HashingVectorizer(stop_words='english', alternate_sign=False,
                                   n_features=opts.n_features)
    X_train = vectorizer.transform(data_train.data)
else:
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                 stop_words='english')
    X_train = vectorizer.fit_transform(data_train.data)
duration = time() - t0
print("Done in {0}s at {1}MB/s".format(duration, data_train_size_mb / duration))
#Printing the dimensions of numpy array
(x,y)=X_train.shape
print("No. of Samples : {0}, No. of Features: {1}".format(x,y))



print("Extracting features from the test data using the same vectorizer")
t0 = time()
X_test = vectorizer.transform(data_test.data)
duration = time() - t0
print("Done in {0}s at {1}MB/s" .format(duration, data_test_size_mb / duration))
#Printing the dimensions of numpy array
(x,y)=X_test.shape
print("No. of Samples: {0}, No. of Features: {1}".format(x,y))



# mapping from integer feature name to original token string
if opts.use_hashing:
    feature_names = None
else:
    feature_names = vectorizer.get_feature_names()


#converting feature names to numpy array
if feature_names:
    feature_names = np.asarray(feature_names)



# Function for a general Classifier
def classifier(clf):
    print('_' * 50)
    print("Training: ")
    print(clf)
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    print("Train time: {0}s".format(train_time))

    t0 = time()
    pred = clf.predict(X_test)
    test_time = time() - t0
    print("Test time:  {0}s".format(test_time))

    score = metrics.accuracy_score(y_test, pred)
    print("accuracy:   {0}".format(score))
    try :
        if feature_names is not None:
            print("top 10 keywords per class:")
            for i, label in enumerate(target_names):
                top10 = np.argsort(clf.coef_[i])[-10:]
                print("{0} \n {1}".format(label, " ".join(feature_names[top10])))


        print("classification report:")
        print(metrics.classification_report(y_test, pred,
                                                    target_names=target_names))
    except AttributeError :
        print("Top features cannot be Selected ")

    print("confusion matrix:")
    print(metrics.confusion_matrix(y_test, pred))
    clf_descr = str(clf).split('(')[0]
    return clf_descr, score, train_time, test_time

results = []
for clf, name in (

        (Perceptron(n_iter=50), "Perceptron"),
        (MultinomialNB(alpha=.01),"Naive Baye's assuming Multinomial Distribution"),
        (BernoulliNB(alpha=.01),"Naive Baye's assuming Bernoulli Distribution"),
        (KNeighborsClassifier(n_neighbors=10), "K Nearest Neighbours"),
         ):
    print('=' * 80)
    print(name)
    results.append(classifier(clf))


# Plotting Results for analysis

indices = np.arange(len(results))

results = [[x[i] for x in results] for i in range(4)]

clf_names, score, training_time, test_time = results
training_time = np.array(training_time) / np.max(training_time)
test_time = np.array(test_time) / np.max(test_time)

plt.figure(figsize=(10, 6))
plt.title("Results !!!")
plt.barh(indices, score, .2, label="score", color='red')
plt.barh(indices + .3, training_time, .2, label="training time",
         color='yellow')
plt.barh(indices + .6, test_time, .2, label="test time", color='darkgreen')
plt.yticks(())
plt.legend(loc='best')
plt.subplots_adjust(left=.25)
plt.subplots_adjust(top=.95)
plt.subplots_adjust(bottom=.05)

for i, c in zip(indices, clf_names):
    plt.text(-.3, i, c)

plt.show()
