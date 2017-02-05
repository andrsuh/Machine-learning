import math
import numpy as np
import random

from operator import itemgetter
from sklearn import metrics
from tabulate import tabulate


class LDA:
    def __init__(self, l=1.0):
        self.L = l

    def fit(self, X, y):
        n, m = X.shape
        classes = y.max() + 1
        expected = []

        for cls in range(classes):
            X_xls = X[np.where(y == cls)]
            expected.append(np.matrix(X_xls.mean(axis=0).reshape(m, 1)))  # column-vector

        cov = np.zeros((m, m))
        for x, label in zip(X, y):
            x = x.reshape(m, 1)
            e = expected[label]
            cov += (x - e) * (x - e).T
        cov /= n
        inv_cov = np.linalg.inv(cov)

        self.coeff = []
        for cls in range(classes):
            class_prob = (y == cls).sum() / n
            alpha = inv_cov * expected[cls]
            beta = -0.5 * expected[cls].T * inv_cov * expected[cls] \
                   + self.L * math.log1p(class_prob)
            self.coeff.append((alpha, beta))

    def predict(self, X):
        result = []
        for x in X:
            class_prob = []
            for cls in range(2):
                a, b = self.coeff[cls]
                prob = x * a + b
                class_prob.append(prob)
            result.append(0 if class_prob[0] > class_prob[1] else 1)
        return np.array(result)


def train_test_split(X, y, n, ratio=0.9):
    for i in range(n):
        mask = np.random.binomial(1, ratio, len(X)) == 1
        yield X[mask], y[mask], X[~mask], y[~mask]


def cv(X, y, classifier, times=10):
    auc = 0.0
    for x_test, y_test, x_train, y_train in train_test_split(X, y, times):
        classifier.fit(x_train, y_train)
        predicted = classifier.predict(x_test)

        tp = ((predicted == 1) * (y_test == 1)).sum()
        fp = ((predicted == 1) * (y_test == 0)).sum()
        tn = ((predicted == 0) * (y_test == 0)).sum()
        fn = ((predicted == 0) * (y_test == 1)).sum()

        # fpr, tpr, thresholds = metrics.roc_curve(y_test, predicted)
        # auc += metrics.auc(fpr, tpr)

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        accuracy = (tp + tn) / (tp + fn + fp + tn)
        f1 = 2 * (precision * recall) / (precision + recall)

        acc = [[precision, recall, accuracy, f1]]
        print(tabulate(acc, headers=["precision", "recall",
                                     "accuracy", "F1-measure"], floatfmt=".2f"))
        # return F1 / times


def normalize(X):
    means = X.mean(axis=0)
    std = X.std(axis=0)
    for sample in X:
        sample -= means
        sample /= std
