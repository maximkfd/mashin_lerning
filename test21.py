import math

import numpy as np
import pylab as pl
from matplotlib.colors import ListedColormap

nItemsInClass = 0
nClasses = 2


# Train data generator
def generateData():
    data = []
    with open('data.txt', 'r') as f:
        for line in f:
            subs = line.split(",")
            x = float(subs[0])
            y = float(subs[1])
            point_class = int(subs[2])
            data.append([[x, y], point_class])
        f.close()
    nItemsInClass = len(data) / 2
    return data


# Separate N data elements in two parts:
# test data with N*testPercent elements
# train_data with N*(1.0 - testPercent) elements
def splitTrainTest(data):
    trainData = []
    testData = []
    test = False
    for row in data:
        if test:
            testData.append(row)
        else:
            trainData.append(row)
        test = not test
    return trainData, testData


# Main classification procedure
def classifyKNN(trainData, testData, k):
    # Euclidean distance between 2-dimensional point
    def dist(a, b):
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    testLabels = []
    for testPoint in testData:
        # Claculate distances between test point and all of the train points
        testDist = [[dist(testPoint, trainData[i][0]), trainData[i][1]] for i in range(len(trainData))]
        # How many points of each class among nearest K
        stat = [0 for i in range(nClasses)]
        for d in sorted(testDist)[0:k]:
            stat[d[1]] += 1
        # Assign a class with the most number of occurences among K nearest neighbours
        testLabels.append(sorted(zip(stat, range(nClasses)), reverse=True)[0][1])
    return testLabels


# Calculate classification accuracy
def calculateAccuracy(k):
    data = generateData()
    trainData, testDataWithLabels = splitTrainTest(data)
    testData = [testDataWithLabels[i][0] for i in range(len(testDataWithLabels))]
    testDataLabels = classifyKNN(trainData, testData, k)
    print("Accuracy: ",
          sum([int(testDataLabels[i] == testDataWithLabels[i][1]) for i in range(len(testDataWithLabels))]) / float(
              len(testDataWithLabels)))


# Visualize classification regions
def showDataOnMesh(k):
    # Generate a mesh of nodes that covers all train cases
    def generateTestMesh(trainData):
        border_offset = 0.5
        x_min = min([trainData[i][0][0] for i in range(len(trainData))]) - border_offset
        x_max = max([trainData[i][0][0] for i in range(len(trainData))]) + border_offset
        y_min = min([trainData[i][0][1] for i in range(len(trainData))]) - border_offset
        y_max = max([trainData[i][0][1] for i in range(len(trainData))]) + border_offset
        h = 0.1
        testX, testY = np.meshgrid(np.arange(x_min, x_max, h),
                                   np.arange(y_min, y_max, h))
        return [testX, testY]

    trainData = generateData()
    testMesh = generateTestMesh(trainData)
    testMeshLabels = classifyKNN(trainData, zip(testMesh[0].ravel(), testMesh[1].ravel()), k)
    classColormap = ListedColormap(['#FF9900', '#00FF00'])
    testColormap = ListedColormap(['#FFCCAA', '#AAFFAA'])
    pl.ion()
    pl.pcolormesh(testMesh[0],
                  testMesh[1],
                  np.asarray(testMeshLabels).reshape(testMesh[0].shape),
                  cmap=testColormap)
    pl.scatter([trainData[i][0][0] for i in range(len(trainData))],
               [trainData[i][0][1] for i in range(len(trainData))],
               c=[trainData[i][1] for i in range(len(trainData))],
               cmap=classColormap)
    pl.pause(0.05)
    # pl.show()


if __name__ == '__main__':
    k = 3
    calculateAccuracy(k)
    showDataOnMesh(k)
    while True:
        input_key = input()
        print(input_key)
        if input_key == '+':
            k += 2
        elif input_key == '-':
            if k == 1:
                continue
            k -= 2
        else:
            continue
        calculateAccuracy(k)
        showDataOnMesh(k)
