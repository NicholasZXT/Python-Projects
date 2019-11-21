
import numpy as np


# cd Machine-Learning-in-Action\Ch05\

dataSet = np.loadtxt("testSet.txt")
data = dataSet[:, :2]
label = dataSet[:, -1]


def sigmoid(x):
    return 1.0/(1 + np.exp(-x))


def gradientDescent(train, label):
    alpha = 0.001
    maxCycles = 500
    m, n = train.shape
    # 初始化权重
    w = np.ones((n, 1))
    for t in range(maxCycles):



