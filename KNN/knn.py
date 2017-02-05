import math
import numpy as np

from random import shuffle
from operator import itemgetter
from collections import Counter, defaultdict


class KNN:
    def __init__(self, k=5):
        self.K = k

    def fit(self, x, y):
        self.x = x
        self.y = y

    def predict(self, x_test):
        return [self.get_nearest_class(point) for point in x_test]

    def get_nearest_class(self, point):
        def weight(dist):
            return 1 / ((dist + np.finfo(float).eps) ** 2)

        dist = [(distance(point, inst), label)
                for inst, label in zip(self.x, self.y)]
        dist = self.find_k_closest(dist)

        label_weight = defaultdict(float)
        for d, label in dist:
            label_weight[label] += weight(d)

        max_weight, best_label = 0.0, None
        for label in label_weight:
            if label_weight[label] > max_weight:
                max_weight, best_label = label_weight[label], label

        return best_label

    def find_k_closest(self, dist):
        min_dist = []
        for i in range(self.K):
            closest = min(dist, key=itemgetter(0))
            min_dist.append(closest)
            dist.remove(closest)
        return min_dist


def distance(p1, p2, n=2.0):
    return math.sqrt(sum(map(lambda a, b: (a - b) ** n, p1, p2)))
    # return sum(p1 * p2) / (np.linalg.norm(p1) * np.linalg.norm(p2))


def train_test_split(X, y, k=10):
    samples, features = X.shape
    selector = [i for i in range(samples)]
    pivot = samples // 2
    for fold in range(k):
        shuffle(selector)
        test = selector[:pivot]
        train = selector[pivot:]
        yield X[test, :], y[test], X[train, :], y[train]


def report(predicted, y_true):
    a = sum(
        [1 for a, b in zip(predicted, y_true) if a == b]) / float(len(predicted))
    return a


def k_fold_cv(X, y, k=2):
    total_accuracy = 0.0
    for x_test, y_test, x_train, y_train in train_test_split(X, y, k):
        knn = KNN()
        knn.fit(x_train, y_train)
        predicted = knn.predict(x_test)

        total_accuracy += report(predicted, y_test)
    return total_accuracy / k


def the_same_values(v):
    return sum([1 for val in v if val != v[0]]) == 0


def correlation(X, y):
    x_t = X.T
    features, samples = x_t.shape

    res = []
    for i in range(features):
        coeff = np.corrcoef(x_t[i], y)[0][1]  # returns matrix
        if coeff > 0.0:
            res.append((i, abs(coeff)))

    res.sort(key=itemgetter(1), reverse=True)
    return res


def forward_selection(X, y, features):
    quality = 0.0
    useful = []
    for feature in features:
        useful.append(feature)
        a = k_fold_cv(X[:, useful], y)
        if a > quality:
            print("Quality: ", a)
            quality = a
        else:
            useful.remove(feature)
        print("Useful features: ", useful)
