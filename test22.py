from random import shuffle

import numpy as np
import pylab as pl
from matplotlib.colors import ListedColormap

train_data = []
nItemsInClass = []
nClasses = 2
k = 3
p = 2
kernel_index = 0


def generate_data():
    global train_data
    if len(train_data) != 0:
        return train_data
    data = []
    with open('data.txt', 'r') as f:
        for line in f:
            subs = line.split(",")
            x = float(subs[0])
            y = float(subs[1])
            point_class = int(subs[2])
            data.append([[x, y], point_class])
        f.close()
    global nItemsInClass
    nItemsInClass = [len(data) / 2, len(data) / 2]
    shuffle(data)
    return data


def split_train_test(data, parts, chosen=0):
    part_len = int(len(data) / parts)
    first_divider = part_len * chosen - 1
    if first_divider < 0:
        first_divider = 0
    second_divider = first_divider + part_len
    # amount = int((1 - 1 / parts) * len(data))
    train_data = data[:first_divider]
    test_data = data[first_divider:second_divider]
    train_data = train_data + (data[second_divider:])
    return train_data, test_data


# Main classification procedure
def classify_knn(train_data, test_data, k):
    # Euclidean distance between 2-dimensional point
    global nClasses

    def dist(a, b):

        def transform(x1_old, x2_old, y1_old, y2_old):
            x1 = x1_old - 0.2
            x2 = x2_old - 0.2
            y1 = y1_old - 0.2
            y2 = y2_old - 0.2
            z1 = (x1 ** 2 + y1 ** 2) ** (1/2) * 5
            z2 = (x2 ** 2 + y2 ** 2) ** (1/2) * 5
            return x1, x2, y1, y2, z1, z2

        x1_old = a[0]
        x2_old = b[0]
        y1_old = a[1]
        y2_old = b[1]
        z1_old = 0
        z2_old = 0
        x1, x2, y1, y2, z1, z2 = transform(x1_old, x2_old, y1_old, y2_old)
        global p
        return (abs(x1 - x2) ** p + abs(y1 - y2) ** p + abs(z1 - z2) ** p) ** (1. / p)

    def kernel(u):
        global kernel_index
        if kernel_index == 0:
            return 1 / 2  # rectangular
        if kernel_index == 1:
            if 1 - abs(u) is None:
                print(u)
                print(3 / 4 * (1 - u * u))
            return 1 - abs(u)  # triangular
        if kernel_index == 2:
            if 3 / 4 * (1 - u * u) is None:
                print(u)
                print(3 / 4 * (1 - u * u))
            return 3 / 4 * (1 - u * u)  # parabolic

    test_classes = []
    for testPoint in test_data:
        test_dist = [[dist(testPoint, train_data[i][0]), train_data[i][1]] for i in range(len(train_data))]
        # How many points of each class among nearest K
        stat = [0 for i in range(nClasses)]
        sorted_distances = sorted(test_dist)
        for d in sorted_distances[0:k]:
            stat[d[1]] += kernel(d[0] / sorted_distances[k][0])
        # Assign a class with the most number of occurences among K nearest neighbours
        test_classes.append(sorted(zip(stat, range(nClasses)), reverse=True)[0][1])
    return test_classes


# def calculate_total_accuracy(parts):
#     for i in range(parts, 1):
#


def calculate_accuracy(parts):
    global k
    data = generate_data()
    summ_accuracy = 0
    for chosen_part in range(0, parts):
        train_data, expert_data = split_train_test(data, parts, chosen=chosen_part)
        test_data = [expert_data[i][0] for i in range(len(expert_data))]
        classified_data = classify_knn(train_data, test_data, k)
        tp, tn, fp, fn = 0, 0, 0, 0
        for j in range(len(expert_data)):
            if expert_data[j][1] == classified_data[j]:
                if expert_data[j][1] == 1:
                    tp += 1
                else:
                    tn += 1
            else:
                if expert_data[j][1] == 0:
                    fp += 1
                else:
                    fn += 1
        precall = tp / (tp + fn)
        recision = tp / (tp + fp)
        if precall + recision == 0:
            return 0, 0
        f_measure = 2 * (recision * precall) / (recision + precall)
        # accuracy = sum([int(classified_data[i] == expert_data[i][1])
        #                 for i in range(len(expert_data))]) / float(len(expert_data))

        # summ_accuracy += accuracy
    return (tp + tn) / len(expert_data), f_measure


def draw_plane(k, should_draw=True):
    # Generate a mesh of nodes that covers all train cases
    def generate_background_data(trainData):
        border_offset = 0.5
        x_min = min([trainData[i][0][0] for i in range(len(trainData))]) - border_offset
        x_max = max([trainData[i][0][0] for i in range(len(trainData))]) + border_offset
        y_min = min([trainData[i][0][1] for i in range(len(trainData))]) - border_offset
        y_max = max([trainData[i][0][1] for i in range(len(trainData))]) + border_offset
        h = 0.1
        testX, testY = np.meshgrid(np.arange(x_min, x_max, h),
                                   np.arange(y_min, y_max, h))
        return [testX, testY]

    global train_data
    train_data = generate_data()
    test_mesh = generate_background_data(train_data)
    test_mesh_classes = classify_knn(train_data, zip(test_mesh[0].ravel(), test_mesh[1].ravel()), k)
    if should_draw:
        class_colormap = ListedColormap(['#FF9900', '#00FF00'])
        test_colormap = ListedColormap(['#FFCCAA', '#AAFFAA'])
        pl.ion()
        pl.pcolormesh(test_mesh[0],
                      test_mesh[1],
                      np.asarray(test_mesh_classes).reshape(test_mesh[0].shape),
                      cmap=test_colormap)
        # pl.scatter(test_mesh[0],
        #            test_mesh[1],
        #            c=np.asarray(test_mesh_classes).reshape(test_mesh[0].shape),
        #            cmap=test_colormap)
        pl.scatter([train_data[i][0][0] for i in range(len(train_data))],
                   [train_data[i][0][1] for i in range(len(train_data))],
                   c=[train_data[i][1] for i in range(len(train_data))],
                   cmap=class_colormap)
        pl.show()


def definePointClass(xdata, ydata):
    def dist(a, b):
        global p
        return ((a[0] - b[0]) ** p + (a[1] - b[1]) ** p) ** (1. / p)

    global nClasses
    global nItemsInClass
    global train_data

    new_point = [xdata, ydata]
    testDist = [[dist(new_point, train_data[i][0]), train_data[i][1]] for i in range(len(train_data))]
    # How many points of each class among nearest K
    stat = [0 for i in range(nClasses)]
    for d in sorted(testDist)[0:k]:
        stat[d[1]] += 1
    # Assign a class with the most number of occurences among K nearest neighbours
    new_point_class = sorted(zip(stat, range(nClasses)), reverse=True)[0][1]
    nItemsInClass[new_point_class] += 1
    train_data.append([new_point, new_point_class])


def onclick(event):
    global nItemsInClass
    global k
    definePointClass(event.xdata, event.ydata)
    draw_plane(k)


def press(event):
    if event.key == 'up':
        global k
        k += 2
        draw_plane(k)
        return
    if event.key == 'down':
        global k
        k -= 2
        draw_plane(k)
        return


if __name__ == '__main__':
    acc = []
    f_max = 0
    acc_k = 0
    acc_j = 0
    acc_ki = 0
    res = []
    shuffles = 5
    for s in range(0, shuffles):
        res_raw = {"k": [], "ki": [], "j": [], "p": [], "acc": [], "f": []}
        print("shuffle", s)
        for i in range(3, 12, 2):
            k = i
            print("k", k)
            for ki in range(0, 3):
                # draw_plane(i, False)
                for ip in range(2, 4):
                    p = ip
                    for j in range(2, 8):
                        accuracy, f_measure = calculate_accuracy(j)
                        res_raw["k"].append(k)
                        res_raw["ki"].append(ki)
                        res_raw["p"].append(p)
                        res_raw["j"].append(j)
                        res_raw["acc"].append(accuracy)
                        res_raw["f"].append(f_measure)
                        # print(k, kernel_index, j, p, accuracy, f_measure, sep="; ", end=";\n")
                kernel_index += 1
            kernel_index = 0
        res.append(res_raw)
        shuffle(train_data)
    aver = {"k": [], "ki": [], "j": [], "p": [], "acc": [], "f": []}
    for i in range(len(res_raw["k"])):
        aver["k"].append(sum(res[j]["k"][i] for j in range(shuffles)) / shuffles)
        aver["ki"].append(sum(res[j]["ki"][i] for j in range(shuffles)) / shuffles)
        aver["p"].append(sum(res[j]["p"][i] for j in range(shuffles)) / shuffles)
        aver["j"].append(sum(res[j]["j"][i] for j in range(shuffles)) / shuffles)
        aver["acc"].append(sum(res[j]["acc"][i] for j in range(shuffles)) / shuffles)
        aver["f"].append(sum(res[j]["f"][i] for j in range(shuffles)) / shuffles)
        print(aver["k"][i], aver["ki"][i], aver["p"][i], aver["j"][i], aver["acc"][i], aver["f"][i], sep="; ")
    for i in range(len(aver["k"])):
        if aver['f'][i] > f_max:
            f_max = aver['f'][i]
            acc_k = aver['k'][i]
            acc_j = aver['j'][i]
            acc_ki = aver['ki'][i]
            acc_p = aver['p'][i]
    print(acc_k, acc_ki, acc_j, acc_p, f_max, sep="; ")
    # draw_plane(3)
    # pl.pause(0)
