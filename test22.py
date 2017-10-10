import math
from random import shuffle
import numpy as np
import pylab as pl
from matplotlib.colors import ListedColormap

train_data = []
nItemsInClass = []
nClasses = 2
k = 3


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
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    test_classes = []
    for testPoint in test_data:
        test_dist = [[dist(testPoint, train_data[i][0]), train_data[i][1]] for i in range(len(train_data))]
        # How many points of each class among nearest K
        stat = [0 for i in range(nClasses)]
        for d in sorted(test_dist)[0:k]:
            stat[d[1]] += 1
        # Assign a class with the most number of occurences among K nearest neighbours
        test_classes.append(sorted(zip(stat, range(nClasses)), reverse=True)[0][1])
    return test_classes


# def calculate_total_accuracy(parts):
#     for i in range(parts, 1):
#


def calculate_accuracy(parts, t=6):
    global k
    data = generate_data()
    summ_accuracy = 0
    for chosen_part in range(0, parts):
        train_data, test_data_with_classes = split_train_test(data, parts, chosen=chosen_part)
        test_data = [test_data_with_classes[i][0] for i in range(len(test_data_with_classes))]
        test_data_classes = classify_knn(train_data, test_data, k)
        accuracy = sum([int(test_data_classes[i] == test_data_with_classes[i][1]) for i in
                    range(len(test_data_with_classes))]) / float(len(test_data_with_classes))
        summ_accuracy += accuracy
    print("Accuracy: ",
          summ_accuracy/parts)
    return summ_accuracy/parts


def draw_plane(k):
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
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

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
    figure = pl.figure()
    figure.canvas.mpl_connect('button_press_event', onclick)
    figure.canvas.mpl_connect('key_press_event', press)
    # draw_plane(k)
    accs = []
    # for k in range(3, 14, 2):
    #     accs.append(calculate_accuracy(10))
    #     pl.pause(0.05)
    #     draw_plane(k)
    draw_plane(7)
    calculate_accuracy(10)
    # while True:
    #     pl.pause(0.05)
    # for i in range(2, 20):
    #     accs.append(calculate_accuracy(i))
    # pl.plot([2*i + 3 for i in range(len(accs))],
    #            accs, c='#FFF000')

    while True:
        pl.pause(0.05)
