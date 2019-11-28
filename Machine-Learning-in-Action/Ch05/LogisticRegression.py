
import numpy as np


# cd 'Machine-Learning-in-Action/Ch05/'

dataSet = np.loadtxt("testSet.txt")
X = dataSet[:, :2]
# y切片出来之后是一个一维的array，还需要转成二维的array
y = dataSet[:, -1].reshape((-1,1))


def sigmoid(x):
    return 1.0/(1 + np.exp(-x))


def gradientAscent(X, y):
    # 数据矩阵X需要增加一列全为1，表示截距
    intercep = np.ones((X.shape[0],1))
    X = np.hstack((intercep,X))
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
    for k in range(maxCycles):
        h = sigmoid(X*w)
        error = y - h
        w = w + alpha*X.transpose()*error
    return w


def gradientAscentCrossEntropy(X,y):
    """
    这个是用交叉熵损失函数表示
    :param X:
    :param y: 这里要求 y 使用{1,-1}来表示类标签，而不是{1,0}
    :return:
    """
    # 数据矩阵X需要增加一列全为1，表示截距
    intercep = np.ones((X.shape[0], 1))
    X = np.hstack((intercep, X))
    # 将 y 里的 0 换成-1
    y = np.where(y > 0,y,-1)
    # 需要将x和y转换为矩阵，以便直接使用矩阵的乘法
    X = np.mat(X)
    # 需要注意的是，这里y还要转成对角矩阵用于向量化迭代
    # y本身是二维的,shape为(n,1)，需要压平成一维array，否则diag函数提取的就是对角线的元素
    y = np.diag(y.flatten())
    y = np.mat(y)
    # 迭代的步长
    alpha = 0.001
    # 最多迭代的次数，这里是写死的
    maxCycles = 500
    n, m = X.shape
    # 初始化权重
    w = np.ones((m, 1))
    for i in range(maxCycles):
        h = sigmoid(-y*X*w)
        w = w + alpha*X.transpose()*y*h
    return w


# 可以看出，这两个结果是一样的
w1 = gradientAscent(X,y)
w2 = gradientAscentCrossEntropy(X,y)


