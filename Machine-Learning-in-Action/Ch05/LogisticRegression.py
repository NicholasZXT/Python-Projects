
import numpy as np


# cd 'Machine-Learning-in-Action/Ch05/'

dataSet = np.loadtxt("testSet.txt")
X = dataSet[:, :2]
y = dataSet[:, -1]


def sigmoid(x):
    return 1.0/(1 + np.exp(-x))


def gradientDescent(X, y):
    # 需要将X和y转换为矩阵，以便直接使用矩阵的乘法
    X = np.mat(X)
    y = np.mat(y)
    # 迭代的步长
    alpha = 0.001
    # 最多迭代的次数，这里是写死的
    maxCycles = 500
    n, m = X.shape
    # 初始化权重
    w = np.ones((m, 1))
    # 开始迭代
    for t in range(maxCycles):
        h = sigmoid(X*w)
        error = y - h
        w = w + alpha *X.transpose()*error
    return w

w = gradientDescent(X,y)

t=X*w


