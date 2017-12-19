#!/usr/bin/env python

import itertools
import logging

import argh
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

from kernels import Kernel
from svm import SVMTrainer


def generate_data():
    data = []
    labels = []
    with open('data.txt', 'r') as f:
        for line in f:
            subs = line.split(",")
            x = float(subs[0])
            y = float(subs[1])
            point_class = int(subs[2])
            if point_class == 0:
                point_class = 1.
            else:
                point_class = -1.
            data.append([x, y])
            labels.append(point_class)
        f.close()
    return np.matrix(data), np.matrix(labels).transpose()


def split_data(data, parts, chosen=0):
    train, test = split_train_test(data, parts, chosen)
    hsplit_train = np.hsplit(train, np.array([2]))
    hsplit_test = np.hsplit(test, np.array([2]))
    return hsplit_train[0], hsplit_train[1], hsplit_test[0], hsplit_test[1]


def split_train_test(data, parts, chosen=0):
    part_len = int(len(data) / parts)
    first_divider = part_len * chosen - 1
    if first_divider < 0:
        first_divider = 0
    second_divider = first_divider + part_len
    # amount = int((1 - 1 / parts) * len(data))
    train_data = data[:first_divider]
    test_data = data[first_divider:second_divider]
    if len(train_data) == 0:
        train_data = data[second_divider:]
    else:
        train_data = np.vstack((train_data, (data[second_divider:])))
    return train_data, test_data



def example(grid_size=30):
    samples, labels = generate_data()
    # samples = np.matrix(np.random.normal(size=num_samples * num_features).reshape(num_samples, num_features))
    # labels = 2 * (samples.sum(axis=1) > 0) - 1.0
    shufled_data = np.concatenate((samples, labels), axis=1)
    np.random.shuffle(shufled_data)
    f1_scores = []
    parts = 5
    trainer = SVMTrainer(Kernel.gaussian(0.16), 0.1)
    for i in range(parts):
        train_data, train_labels, test_data, test_labels = split_data(shufled_data, parts, i)
        predictor = trainer.train(train_data, train_labels)

        f1 = f_measure(test_labels, predictor, test_data)
        print(f1)
        f1_scores.append(f1)
    print(np.average(np.array(f1_scores)))
    predictor = trainer.train(samples, labels)
    plot(predictor, samples, labels, grid_size)


def f_measure(labels, predictor, samples):
    tp = 0.001
    tn = 0
    fn = 0
    fp = 0
    for i in range(len(samples)):
        sample = samples[i]
        label = labels[i][0][0]
        prediction = predictor.predict(sample)
        if prediction == label:
            if prediction == 1:
                tp += 1
            else:
                tn += 1
        else:
            if prediction == 1:
                fp += 1
            else:
                fn += 1
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    return 2 * precision * recall / (precision + recall)


def plot(predictor, X, y, grid_size):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_size),
                         np.linspace(y_min, y_max, grid_size),
                         indexing='ij')
    flatten = lambda m: np.array(m).reshape(-1, )

    result = []
    for (i, j) in itertools.product(range(grid_size), range(grid_size)):
        point = np.array([xx[i, j], yy[i, j]]).reshape(1, 2)
        result.append(predictor.predict(point))

    Z = np.array(result).reshape(xx.shape)

    plt.contourf(xx, yy, Z,
                 cmap=cm.Paired,
                 levels=[-0.001, 0.001],
                 extend='both',
                 alpha=0.8)
    plt.scatter(flatten(X[:, 0]), flatten(X[:, 1]),
                c=flatten(y), cmap=cm.Paired)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.show()


if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)
    argh.dispatch_command(example)
