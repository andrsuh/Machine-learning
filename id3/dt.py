import math
import numpy as np
import random

from collections import Counter
from operator import itemgetter
from sklearn import metrics
from tabulate import tabulate


def jinny_quality(feature, label, pred):
    def quality(part):
        if len(part) <= 0:
            return float("inf")
        fst_prob = (part == 1).sum() / len(part)
        snd_prob = (part == 0).sum() / len(part)
        return ((fst_prob * (1 - fst_prob) + snd_prob * (1 - snd_prob))) * len(part)

    fst_part = label[pred(feature) == True]
    snd_part = label[pred(feature) == False]

    return quality(fst_part) / len(feature) + quality(snd_part) / len(feature)


def get_feature_values(f_num, feature):
    if f_types[f_num] == "discr":
        c = Counter(feature)
        return [(k, lambda f: f == k) for k in c.keys()]
    values = np.random.uniform(feature.min(), feature.max(), 30)
    for value in values:
        yield value, lambda f: f >= value


def separate(f_num, feature, label):
    b_quality, b_threshold = float("inf"), None
    for threshold, pred in get_feature_values(f_num, feature):
        quality = jinny_quality(feature, label, pred)
        if quality < b_quality:
            # print(quality)
            b_quality, b_threshold = quality, threshold
    pred = lambda f: f[f_num] >= b_threshold if f_types[f_num] == "contin" else lambda f: f[f_num] == b_threshold
    return b_quality, pred


def find_best_predicate(X, y):
    b_quality, b_feature, b_pred = float("inf"), None, None
    for feature in range(X.shape[1]):
        quality, pred = separate(feature, X[:, feature], y)
        if quality < b_quality:
            b_quality, b_feature, b_pred = quality, feature, pred

    return b_pred


class Node:
    def __init__(self, label, pred=None, left=None, right=None):
        self.pred = pred
        self.left = left
        self.right = right
        self.label = label


class Decision3:
    def fit(self, X, y):
        self.tree = self.learn_id3(X, y)
        return self

    def learn_id3(self, X, y):
        if not len(X[y == 1]):
            return Node(0)
        elif not len(X[y == 0]):
            return Node(1)

        pred = find_best_predicate(X, y)

        if pred:
            positive = np.array([pred(sample) for sample in X])
            feature_1, label_1 = X[positive], y[positive]
            feature_2, label_2 = X[~positive], y[~positive]
        else:
            feature_1, label_1, feature_2, label_2 = X, y, [], []

        if len(feature_1) == 0 or len(feature_1) < 0.01 * len(X):
            major_class = 1 if label_2.sum() >= len(label_2) / 2 else 0
            return Node(major_class)

        if len(feature_2) == 0 or len(feature_2) < 0.01 * len(X):
            major_class = 1 if label_1.sum() >= len(label_1) / 2 else 0
            return Node(major_class)

        return Node(None, pred, self.learn_id3(feature_1, label_1),
                    self.learn_id3(feature_2, label_2))

    def predict(self, X):
        predict = np.zeros(X.shape[0], dtype="int")
        for i, sample in enumerate(X):
            tree = self.tree
            while (True):
                if tree.label is not None:
                    predict[i] = tree.label
                    break
                if tree.pred(sample):
                    tree = tree.left
                else:
                    tree = tree.right
        return predict


def train_test_split(X, y, n, ratio=0.5):
    for i in range(n):
        mask = np.random.binomial(1, ratio, len(X)) == 1
        yield X[mask], y[mask], X[~mask], y[~mask]


def cv(X, y, classifier, times=10):
    auc_total = 0.0
    f1_total = 0.0
    for x_test, y_test, x_train, y_train in train_test_split(X, y, times):
        classifier.fit(x_train, y_train)
        predicted = classifier.predict(x_test)

        tp = ((predicted == 1) * (y_test == 1)).sum()
        fp = ((predicted == 1) * (y_test == 0)).sum()
        tn = ((predicted == 0) * (y_test == 0)).sum()
        fn = ((predicted == 0) * (y_test == 1)).sum()

        fpr, tpr, thresholds = metrics.roc_curve(y_test, predicted)
        auc = metrics.auc(fpr, tpr)
        auc_total += auc

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        accuracy = (tp + tn) / (tp + fn + fp + tn)
        F1 = 2 * (precision * recall) / (precision + recall)
        f1_total += F1

        acc = [[precision, recall, accuracy, auc, F1]]
        print(tabulate(acc, headers=["precision", "recall",
                                     "accuracy", "AUC", "F1-measure"], floatfmt=".2f"))
    return f1_total / times


def normalize(X):
    means = X.mean(axis=0)
    std = X.std(axis=0)
    for sample in X:
        sample -= means
        sample /= std


def determine_features_types(X):
    f_types = {}
    for i in range(X.shape[1]):
        c = Counter(X[:, i])
        f_types[i] = "discr" if len(c) < 5 else "contin"
    return f_types
