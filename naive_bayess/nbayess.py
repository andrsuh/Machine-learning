#!/usr/bin/python3

import math
import time

import numpy as np
from collections import defaultdict
from pathlib import Path
from tabulate import tabulate

dataset = defaultdict(list)

def iter_dirs():
    for i in range(10):
        iter_docs("part{}".format(i + 1))


def iter_docs(dirname):
    for path in Path(dirname).iterdir():
        if path.is_file():
            dataset[dirname].append(
                (form_doc(path.open()), 0 if "spmsg" in path.name else 1)
            )


def form_doc(file):
    return [int(word) for line in file
            for word in line.strip("Subject: \n").split()]


class Vectorizer:
    def __init__(self):
        self._vocabulary = set()

    def fit(self, docs):
        for doc in docs:
            for word in doc:
                self._vocabulary.add(word)

        self._demension = max(self._vocabulary)
        return self

    def transform(self, docs):
        matrix = []
        for doc in docs:
            vector = [0] * (self._demension + 1)
            for word in doc:
                if word in self._vocabulary:
                    vector[word] += 1
                else:
                    vector[len(vector) - 1] += 1
            matrix.append(vector)
        return matrix

    def fit_transform(self, docs):
        return self.fit(docs).transform(docs)


class NaiveBayes:
    def fit(self, X, y):
        n_docs, n_words = len(X), len(X[0])
        n_classes = 2

        epsilon = np.finfo(float).eps
        class_prob = [.5] * n_classes
        feature_prob = [None] * n_classes
        for c in range(n_classes):
            class_prob[c] = y.count(c)
            feature_prob[c] = [
                sum(X[d][i] * (y[d] == c) for d in range(n_docs)) + epsilon
                for i in range(n_words)
                ]

        class_prob = [prob / sum(class_prob) for prob in class_prob]
        for i in range(n_words):
            total_prob = sum(feature_prob[c][i] for c in range(n_classes))
            for c in range(n_classes):
                feature_prob[c][i] /= total_prob

        self._class_prob = class_prob
        self._feature_prob = feature_prob
        return self

    def predict(self, X):
        n_classes = 2

        y = []
        lp = {}
        for doc in X:
            for c in range(n_classes):
                log_prob = \
                    sum(count * math.log(prob)
                        for count, prob in zip(doc, self._feature_prob[c]))
                log_prob += math.log(self._class_prob[c])
                lp[c] = log_prob
            y.append(1 if lp[0] / lp[1] > 1.4 else 0)
        return y


def rotate(lst, n=1):
    return lst[n:] + lst[:n]


def train_test_split(k=10):
    fold = [i for i in range(10)]
    key = "part{}"

    for i in range(k):
        x_test, y_test, x_train, y_train = [], [], [], []

        for ind in fold:
            for doc, label in dataset[key.format(ind + 1)]:
                # print(doc)
                if ind == fold[0]:
                    x_test.append(doc)
                    y_test.append(label)
                else:
                    x_train.append(doc)
                    y_train.append(label)

        fold = rotate(fold)
        yield x_test, y_test, x_train, y_train


def k_fold_cv(k=10):
    v = Vectorizer()
    F1_total = 0.0

    for x_test, y_test, x_train, y_train in train_test_split(k):
        X_train = v.fit_transform(x_train)
        X_test = v.transform(x_test)

        clf = NaiveBayes().fit(X_train, y_train)
        y_pred = np.array(clf.predict(X_test))
        F1_total += classification_report(y_test, y_pred, ["spam", "non_spam"])
        print("Distance between predicted and correct vectors: ",
              sum([1 for t, p in zip(y_test, y_pred) if t != p]))

    print("Total F1 measure result: ", F1_total / 10.0)


def precision_recall_total(y_true, y_pred):
    tp = (y_pred * y_true).sum()
    fp = (y_pred * ~y_true).sum()
    fn = (~y_pred * y_true).sum()
    tn = (~y_pred * ~y_true).sum()

    # print(tabulate(
    #         [["tp {}".format(tp),"fp {}".format(fp)],
    #          ["fn {}".format(fn), "tn {}".format(tn)]],
    #         headers=["", "", "", "",], floatfmt=".2f"))


    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    accuracy = (tp + tn) / (tp + fn + fp + tn)
    F1 = 2 * (precision * recall) / (precision + recall)

    return precision, recall, accuracy, F1, int(y_true.sum())


def classification_report(y_true, y_pred, labels):
    acc = []
    F1_total = 0.0

    for c, label in enumerate(labels):
        precision, recall, accuracy, F1, total = precision_recall_total(
            np.asarray(y_true) == c, np.asarray(y_pred) == c)
        F1_total += F1
        acc.append([label, precision, recall, accuracy, F1, total])

    print(tabulate(acc, headers=["", "precision", "recall",
                                 "accuracy", "F1-measure", "total"], floatfmt=".2f"))
    return F1_total / 2.0


if __name__ == "__main__":
    iter_dirs()
    k_fold_cv()
