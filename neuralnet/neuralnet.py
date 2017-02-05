#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from tabulate import tabulate
from operator import itemgetter


def logistic(x, der=False):
    if der:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))


class NN:
    def __init__(self, f_num, hl_size, ol_size=1):
        self.f_num = f_num
        self.hl_size = hl_size
        self.ol_size = ol_size

    def initialize(self):
        self.syn0 = 2 * np.random.random((self.f_num, self.hl_size)) - 1
        self.syn1 = 2 * np.random.random((self.hl_size, self.ol_size)) - 1

    def fit(self, X, y):
        y = np.array(np.matrix(y).T)

        self.initialize()

        for it in range(1000000):
            selector = np.random.randint(0, X.shape[0], 1)
            l0 = X[selector]
            l1 = logistic(np.dot(l0, self.syn0))
            l2 = logistic(np.dot(l1, self.syn1))

            l2_error = (y[selector] - l2)

            # cross entropy
            # yt = y[selector][0][0]
            # a = l2[0][0]
            # l2_error = [[(yt / a + np.finfo(float).eps)
            #                 - ((1 - yt) / (1 - a + np.finfo(float).eps))]]
            #

            if (it % 10000) == 0:
                print("Error:" + str(np.mean(np.abs(l2_error))))

            l2_delta = l2_error * logistic(l2, True)
            l1_error = l2_delta.dot(self.syn1.T)
            l1_delta = l1_error * logistic(l1, True)
            prob_1 = np.random.random()
            prob_2 = np.random.random()
            step = it / 100000
            self.syn1 += (step if prob_1 < 0.98 else 8.0) * l1.T.dot(l2_delta)
            self.syn0 += (step if prob_2 < 0.98 else 8.0) * l0.T.dot(l1_delta)

    def predict(self, X):
        l1 = logistic(np.dot(X, self.syn0))
        return logistic(np.dot(l1, self.syn1)).round()


def correlation(X, y):
    x_t = X.T
    features, samples = x_t.shape
    res = []
    for i in range(features):
        coeff = np.corrcoef(x_t[i], y)[0][1]  # returns matrix
        if abs(coeff) > 0.0:
            res.append((i, abs(coeff)))

    res.sort(key=itemgetter(1), reverse=True)
    return res


def forward_selection(X, features):
    using = []
    quality = 0.0
    for f in features:
        using.append(f)
        q = cv(X[:, using], y, NN(len(using), 15, 1))
        if q > quality:
            print("Quality: {} Using: {}".format(q, using))
            quality = q
        else:
            using.pop()
    return using


def train_test_split(X, y, n, ratio=0.5):
    for i in range(n):
        mask = np.random.binomial(1, ratio, len(X)) == 1
        yield X[mask], y[mask], X[~mask], y[~mask]


def get_metrics(predicted, y_test):
    tp = ((predicted == 1) * (y_test == 1)).sum()
    fp = ((predicted == 1) * (y_test == 0)).sum()
    tn = ((predicted == 0) * (y_test == 0)).sum()
    fn = ((predicted == 0) * (y_test == 1)).sum()
    fpr, tpr, thresholds = metrics.roc_curve(y_test, predicted)
    auc = metrics.auc(fpr, tpr)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    accuracy = (tp + tn) / (tp + fn + fp + tn)
    F1 = 2 * (precision * recall) / (precision + recall)
    return precision, recall, accuracy, F1, auc


def cv(X, y, classifier, times=1):
    auc_total = 0.0
    f1_total = 0.0
    for x_test, y_test, x_train, y_train in train_test_split(X, y, times):
        classifier.fit(x_train, y_train.T)

        predicted = classifier.predict(x_test)
        print(predicted)
        precision, recall, accuracy, F1, auc = get_metrics(predicted, y_test)

        auc_total += auc
        f1_total += F1

        acc = [[precision, recall, accuracy, F1, auc]]
        print(tabulate(acc, headers=["precision", "recall",
                                     "accuracy", "F1-measure", "AUCROC"], floatfmt=".2f"))
    return f1_total / times


def normalize(X):
    means = X.mean(axis=0)
    std = X.std(axis=0)
    for sample in X:
        sample -= means
        sample /= std