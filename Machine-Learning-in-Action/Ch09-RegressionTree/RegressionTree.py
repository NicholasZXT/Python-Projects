
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# %cd Machine-Learning-in-Action/Ch09-RegressionTree
# cd .\Machine-Learning-in-Action\Ch09-RegressionTree\



"""
这里创建的回归树是使用的是一个字典来保存，和ID3算法里不同的是，这里的字典里有四个项目：
1. 切分的特征
2. 切分特征的取值
3. 左子树
4. 右子树
"""


def binarySplitData(data, y, featureIndex, value):
    """
    用于根据给定的feature(用Index表示)和相应的value来二分数据集
    :param data: np.array类型
    :param y: 类标签或者值
    :param featureIndex: 特征在data中的列数
    :param value: 用于切分的特征值
    :return: subData0, subData1, suby0, suby1
    """
    subData0 = data[data[:, featureIndex] <= value]
    suby0 = y[data[:, featureIndex] <= value]
    subData1 = data[data[:, featureIndex] > value]
    suby1 = y[data[:, featureIndex] > value]
    return subData0, subData1, suby0, suby1


def calRegressionLeaf(data, y):
    """
    只适用于叶子节点
    这个函数用于计算回归树里每个叶子节点的值——也就是叶子节点内所有值的均值
    其实不需要用到data，只需要用到y
    :param data:
    :param y:
    :return:
    """
    return np.mean(y)


def regressionError(data, y):
    """
    计算数据集data对应的误差
    :param data:
    :param y:
    :return:
    """
    return np.var(y) * len(y)


def chooseBestSplit(data, y, calLeafValue=calRegressionLeaf, calError=regressionError, stopOptions=(1,4)):
    """
    这个函数是回归树里最复杂、也是最重要的部分。
    它遍历data中所有的特征以及可能的取值（data中出现过的值），找到最佳的切分特征和对应的切分value；
    如果找不到“好”的切分，就直接返回一个叶子节点。
    有三种情况会停止切分数据集data，直接创建叶子节点：
    1. data对应的y的值都一样，此时就不需要分割数据，将当前data作为叶子节点，直接返回叶子节点的值
    2. 如果切分后误差降低不明显（小于tolenrance)，也不做切分，将当前data作为叶子节点，直接返回叶子节点的值
    3. 如果切分后，某个子集含有的样本数小于指定值（minSample)，n也不做切分，将当前data作为叶子节点，直接返回叶子节点的值
    其实2和3属于预剪枝。
    :param data:
    :param y:
    :param calLeafValue:
    :param calError:
    :param stopOptions:
    :return: 返回的是用于切分的bestFeature(用index表示）以及相应的切分value；
    如果不做切分，那么返回的feature就是None，返回的值就是叶子节点的值
    """
    tolenrance = stopOptions[0]  # 允许的误差下降值
    minSamples = stopOptions[1]  # 切分的最少样本数
    # 情况1
    if len(set(y)) == 1:
        return None, calLeafValue(data, y)
    n, m = data.shape
    # 计算当前数据集的误差
    baseError = calError(data, y)
    # 初始化最佳特征的index，最佳切分value以及最佳切分的误差
    bestFeatureIndex = 0
    bestSplitValue = 0
    bestError = np.inf
    # 遍历每个特征
    for featureIndex in range(m):
        # 对于给定的（连续）特征，候选的切分点就是data中该特征出现过的所有值！！！
        for splitValue in set(data[:, featureIndex]):
            # 用给定的特征和切分的value，二分数据集
            subData0, subData1, suby0, suby1 = binarySplitData(data, y, featureIndex, splitValue)
            # 如果划分的数据集太小，那就跳过这个splitValue
            if (subData0.shape[0] < minSamples or subData1.shape[0] < minSamples): continue
            # 计算划分后两个数据集的误差和
            newError = calError(subData0, suby0) + calError(subData1, suby1)
            # 如果新的误差小于已有的bestError，就更新最佳特征和最佳切分点，同时更新最小误差
            if newError < bestError:
                bestFeatureIndex = featureIndex
                bestSplitValue = splitValue
                bestError = newError
    # 上面两个for循环结束，就能得到所有特征和对应切分点下最小的误差
    # 比较误差下降的程度是否够大，不够大就不进行切分，直接将data作为叶子节点，并返回叶子节点的值
    if (baseError - bestError) < tolenrance:
        return None, calLeafValue(data, y)
    # 下面这个判断似乎是多余的。。。
    # 使用最佳特征和切分点对数据集进行分割
    subData0, subData1, suby0, suby1 = binarySplitData(data, y, bestFeatureIndex, bestSplitValue)
    if (subData0.shape[0] < minSamples or subData1.shape[0] < minSamples):
        return None, calLeafValue(data, y)
    return bestFeatureIndex, bestSplitValue


def createRegressionTree(data, y, calLeafValue=calRegressionLeaf, calError=regressionError, stopOptions=(1, 4)):
    """
    CART算法，递归创建决策树
    :param data:
    :param y:
    :param calLeafValue:
    :param calError:
    :param stopOptions:
    :return:
    """
    # 首先尝试切分数据，并找到最佳的切分特征和切分点
    bestFeatureIndex, bestSplitValue = chooseBestSplit(data, y,  calLeafValue, calError, stopOptions)
    # 如果不能切分，那就直接返回叶子节点的值
    if bestFeatureIndex == None:
        return bestSplitValue
    # 初始化一棵空树
    tree = {}
    # 添加最佳的切分特征和对应的切分点
    tree["featureIndex"] = bestFeatureIndex
    tree["splitValue"] = bestSplitValue
    # 根据最佳切分特征和切分点来二分数据集
    leftData, rightData , lefty, righty = binarySplitData(data, y, bestFeatureIndex, bestSplitValue)
    # 递归创建左右子树
    tree["left"] = createRegressionTree(leftData, lefty, calLeafValue, calError, stopOptions)
    tree['right'] = createRegressionTree(rightData, righty,  calLeafValue, calError, stopOptions)


if __name__ == "__main__":

    data = pd.read_csv("ex0.txt", sep='\t', header=None).values
    y = data[:, -1]
    data = data[:, :-1]



    data = pd.read_csv("ex00.txt", sep='\t', header=None)
    y = data[:, -1]
    data = data[:, :-1]