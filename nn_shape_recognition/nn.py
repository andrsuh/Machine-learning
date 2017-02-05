import gzip
from operator import itemgetter

import scipy.misc as sss
from PIL import Image
from pylab import *
from scipy.misc.pilutil import *
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score


def logistic(arg, der=False):
    if der:
        return arg * (1 - arg)
    return 1 / (1 + np.exp(-arg))


class NN:
    def __init__(self, f_num, hl_size, ol_size=10):
        self.f_num = f_num
        self.hl_size = hl_size
        self.ol_size = ol_size

    def initialize(self):
        self.syn_weights_0 = 2 * np.random.random((self.f_num, self.hl_size)) - 1
        self.syn_weights_1 = 2 * np.random.random((self.hl_size, self.ol_size)) - 1

    def labels_transform(self, y):
        transformed_y = []
        cls_num = max(y) + 1
        for label in y:
            new_label = np.zeros(cls_num)
            new_label[label] = 1
            transformed_y.append(new_label)
        return np.array(transformed_y)

    def fit(self, X, y):
        y = self.labels_transform(y)
        self.initialize()

        err = 0.0
        count_iter = 300000
        for it in range(count_iter):
            selector = np.random.randint(0, X.shape[0], 3)
            l0 = X[selector]
            l1 = logistic(np.dot(l0, self.syn_weights_0))
            l2 = logistic(np.dot(l1, self.syn_weights_1))

            yt = y[selector]
            l2_error = (yt - l2)

            # cross entropy
            # yt = y[selector][0][0]
            # a = l2[0][0]
            # l2_error = [[(yt / a + np.finfo(float).eps)
            #                 - ((1 - yt) / (1 - a + np.finfo(float).eps))]]

            err += np.mean(np.abs(l2_error))
            if (it % 10000) == 0:
                print("Iter: ", it, " Error:" + str(err / it))

            l2_delta = l2_error * logistic(l2, True)
            l1_error = l2_delta.dot(self.syn_weights_1.T)
            l1_delta = l1_error * logistic(l1, True)

            step = 0.7
            self.syn_weights_1 += step * l1.T.dot(l2_delta)
            self.syn_weights_0 += step * l0.T.dot(l1_delta)

    def predict(self, X):
        prediction = np.zeros(len(X), dtype=int)
        for i, sample in enumerate(X):
            l1 = logistic(np.dot(sample, self.syn_weights_0))
            l2 = logistic(np.dot(l1, self.syn_weights_1))
            prediction[i] = l2.argmax()
        return prediction

    def predict_one(self, sample):
        l1 = logistic(np.dot(sample, self.syn_weights_0))
        l2 = logistic(np.dot(l1, self.syn_weights_1))
        return l2.argmax()


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


def train_test_split(X, y, n, ratio=0.5):
    for i in range(n):
        mask = np.random.binomial(1, ratio, len(X)) == 1
        yield X[mask], y[mask], X[~mask], y[~mask]


def cv(X, y, classifier, times=1):
    f1_total = 0.0
    for x_test, y_test, x_train, y_train in train_test_split(X, y, times):
        classifier.fit(x_train, y_train.T)
        predicted = classifier.predict(x_test)
    return f1_total / times


def load_mnist_images(filename):
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1, 1, 28, 28)
    return data


def load_mnist_labels(filename):
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    return data


def load_dataset(dataset='training'):
    features_number = 28 * 28

    if dataset == "training":
        images = load_mnist_images('train-images-idx3-ubyte.gz')
        labels = load_mnist_labels('train-labels-idx1-ubyte.gz')
    else:
        images = load_mnist_images('t10k-images-idx3-ubyte.gz')
        labels = load_mnist_labels('t10k-labels-idx1-ubyte.gz')

    images = (np.array(images)).reshape(len(images), features_number)
    min_max_scaler = preprocessing.MinMaxScaler()
    images = min_max_scaler.fit_transform(images)

    labels = (np.array(labels, dtype=int)).reshape(len(labels), 1)

    return images, labels


def read_image(file):
    img = Image.open(file).convert('L')
    img = np.asarray(img)
    return img.reshape((1, 784))


X_train, y_train = load_dataset()
X_test, y_test = load_dataset('testing')

X_train = X_train[:50000]
y_train = y_train[:50000]

n = NN(X_train.shape[1], 150, 10)
n.fit(X_train, y_train)

prediction = n.predict(X_test)

false_value = 0
for i, (true, predict) in enumerate(zip(y_test, prediction)):
    if true != predict:
        print('label:', true, ' result: ', predict)

        false_value += 1
print((len(X_test) - false_value) / len(X_test) * 100)

print("F1 measure: ", f1_score(y_test, prediction, average="macro"))
print("Accuracy  : ", accuracy_score(y_test, prediction))

while True:
    print(n.predict_one(read_image("input.png")))
    a = input()
