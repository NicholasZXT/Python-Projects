from math import log
import numpy as np
import operator
import matplotlib.pyplot as plt


def createDataSet():
    """
    这个函数只是用于生成一个测试数据
    返回值：
    dataSet:一个nested list，每一个sublist包括两部分，前面都是对应于feature的取值，1-yes,0-no,最后一个表示class label
    featureNames：list,表示的是sublist中各个feature的名称，因为dataSet中没有存储feature的名称，只存储了值
    """
    # data = [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
    data = np.array([[1, 1], [1, 1], [1, 0], [0, 1], [0, 1]])
    label = np.array(['yes', 'yes', 'no', 'no', 'no'])
    featureNames = ['no surfacing', 'flippers']
    return data, label, featureNames


def shannonEntropy(dataLabel):
    """
    这个函数用于计算当前数据集data的香农熵，而不是直接计算使得information gain最大的feature
    计算数据集的香农熵不需要特征，只需要这个数据集对应的类标签就行了。
    :param dataLabel: np.array对应data每一行观察的类标签，注意，这个类标签是字符，而不是数字
    :return: shannonEnt，返回的是当前数据集的香农熵
    """
    # 首先计算entry的数量
    numEntries = len(dataLabel)
    # 统计dataLabel中类的个数,以及每一类的数量，用dict存储
    classLabels = {}
    for label in dataLabel:
        classLabels[label] = classLabels.get(label, 0) + 1
    # 开始计算信息熵
    shannonEnt = 0
    for classCounts in classLabels.values():
        # prob是每一类的比例，这个比例作为概率分布看待
        prob = float(classCounts) / numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


def splitData(data, label, feature, value):
    """
    用于根据给定的feature以及value划分数据集，返回的是data中feature=value的数据，
    并且 feature 所在的列会被删除
    :param data: np.array格式
    :param label: 类标签
    :param feature: 用于划分的特征在data中的索引，int类型
    :param value: 用于划分data的feature所取的值
    :return: subData，返回的子集，subLabel，对应的类标签
    """
    subData = data[data[:, feature] == value]
    subData = np.delete(subData, feature, 1)
    subLabel = label[data[:, feature] == value]
    return subData, subLabel


def chooseBestFeature(data, label):
    """
    从data中选择最佳分割的feature
    :param data: np.array，
    :param label: 类标签
    :return: bestFeature，最佳分割的feature在data中的索引
    """
    numFeature = data.shape[1]
    # 当前数据集本身的香农熵，其实这个不用计算，因为它对于每个特征都是一样的
    baseEntropy = shannonEntropy(dataLabel=label)
    # 初始化最佳信息增益和最佳feature的索引
    bestInfoGain = 0.0
    bestFeature = -1
    # 循环计算每个特征的分割后的信息增益
    for feature in range(numFeature):
        # 获得该feature所取值的个数
        uniqueValues = set(data[:, feature])
        # 初始化分割后数据集的香农熵
        newEntropy = 0
        # 对于当前feature的每个值，进行分割数据，同时计算每个子集的香农熵，最后得到分割后的香农熵之和
        for value in uniqueValues:
            subData, subLabel = splitData(data, label, feature, value)
            prob = subData.shape[0]/float(data.shape[0])
            newEntropy += prob * shannonEntropy(subLabel)
        # 计算该feature的信息增益
        infoGain = baseEntropy - newEntropy
        # 更新最大的信息增益和对应的feature
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = feature
    return bestFeature


def majorVote(data, label):
    """
    这个函数用于处理特征使用完之后，叶子节点中仍然含有多类的数据，此时该叶子节点的类标签由多数投票决定
    :param data: np.array类型，其实不需要使用data，只需要使用label就行了
    :param label:
    :return: keyMax，返回计数最多的类标签
    """
    labelCount = {}
    for classLabel in label:
        labelCount[classLabel] = labelCount.get(classLabel, 0) + 1
    labelMax = max(labelCount, key=labelCount.get)
    return labelMax


def trainDecisionTree(data, label, featureNames):
    # 递归结束的第一个条件，数据集的所有类标签都属于同一类
    if len(set(label)) == 1:
        return label[0]
    # 递归结束的第二个条件，用完了所有的特征，但是数据集还是含有多个类别，这时使用上面的majorVote给出类标签
    if len(data) == 0:
        return majorVote(data, label)
    bestFeature = chooseBestFeature(data, label)
    bestFeatureName = featureNames[bestFeature]
    tree = {bestFeatureName:{}}
    del featureNames[bestFeature]
    featureValues = set(data[:, bestFeature])
    for value in featureValues:
        subData, subLabel = splitData(data, label, bestFeature, value)
        tree[bestFeatureName][value] = trainDecisionTree(subData, subLabel, featureNames[:])
    return tree


def predictDecisionTree(tree, featureNames, testData):
    rootFeatureName = tree.keys()[0]
    rootFeatureIndex = featureNames.index(rootFeatureName)
    subTree = tree[rootFeatureName]
    for value in subTree.keys():
        if testData[rootFeatureIndex] == value:



# -----
data, label, featureNames = createDataSet()

tree = trainDecisionTree(data, label,featureNames)




def createDecisionTree(dataSet, featureNames):
    """
    递归创建一棵决策树，创建的决策树是以nested dict的形式存在的，
    这个nested dict的每一层key,是feature和feature的value交替进行，value不是类标签，就是子树(sub-dict)
    输入：
    dataSet：是一个nested list，每个sublist代表一个观测，前面是各个feature的取值，最后一个是该观测的class label
    labels：是feature的名称，list
    输出：
    deciTree：构建好的二叉决策树，是一个nested dict，一个dict表示一棵子树，示例如下
    {'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}}
    """
    # 首先将dataSet中所有entries的类标签提取出来，从而知道总共有多少类
    classLabels = [entries[-1] for entries in dataSet]
    # 判断递归是否停止有两个情况：
    # 情况一：当前dataSet中所有的entries都属于同一个类，那就直接返回这个类标签
    if classLabels.count(classLabels[0]) == len(classLabels):
        return classLabels[0]
    # 情况二：当前dataSet中所有的feature都已经被使用过了，那就采取多数表决原则决定类标签（调用上述的majorityVote）
    # 所有feature都被使用过时，dataSet中的sublist就只有一个值——classLabel了，所以长度==1
    if len(dataSet[0]) == 1:
        return majorityVote(dataSet)
    # 上述两个都不满足，说明还可以再进行分割

    # 首先找出当前dataSet下的最佳分割feature(index表示)
    bestFeatureIndex = chooseBestFeature(dataSet)
    # 最佳feature的名称
    bestFeatureName = featureNames[bestFeatureIndex]

    # 使用这个最佳feature构建一棵树，
    # 先用这个feature作为子树的根节点
    deciTree = {bestFeatureName: {}}

    # 删除已经使用过的feature名称
    # 这里要特别注意，不能使用下面这句，因为featureNames是作为list传进来的，这里是pass-by-reference
    # 直接删除会导致外面的list内容也没有了
    # del featureNames[bestFeatureIndex]
    # 应当先获取copy
    featureNames = featureNames[:]
    del featureNames[bestFeatureIndex]

    # 统计最佳feature的取值个数
    featureValues = [entries[bestFeatureIndex] for entries in dataSet]
    uniqueValue = set(featureValues)
    # 对于最佳feature的每个取值，分别进行切割，同时构建子树
    for value in uniqueValue:
        # 根据最佳feature的取值,得到feature=value的训练集合subset
        subset = filterDataset(dataSet, bestFeatureIndex, value)
        # 对于这个subset，递归创建子树，注意，这里使用的featureNames已经去掉了使用过的最佳feature的名称
        deciTree[bestFeatureName][value] = createDecisionTree(subset, featureNames)
    return deciTree



def classifyDecisionTree(deciTree, featureNames, inputList):
    '''
    决策树分类函数
    输入：
    deciTree:已经由createDecisiontree创建好的决策树，nested dict
    fealtureNames:feature的名称，nested list
    inputList:待分类的观测向量,list
    输出：
    classLabels：inputList被分到的类标签
    '''
    # 首先，给定了inputList，需要知道决策树deciTree中用于分割的第一个feature（根节点）是哪个，
    # 这个信息就是deciTree这个nested dict中的key，虽然这个key只有一个，还是需要使用index=0
    rootName = list(deciTree.keys())[0]
    rootIndex = featureNames.index(rootName)
    # 根节点之下的子树就是root这个key对应的value——它也是一个nested dict,
    # 不过它是子树的集合(不是单独的一棵子树)，而是root的每一个value都对应于一棵子树，所以是集合
    subtreeSet = deciTree[rootName]
    # 对于根据root创建的subtreeSet，它的key不再是feature了，而是上一个feature root的不同取值，
    # 每个不同的取值对应于不同的subtree——这个才是单独的一棵子树，它的key只有一个，就是这棵子树的root
    for rootValue in subtreeSet.keys():
        if inputList[rootIndex] == rootValue:
            # 根节点root不同的value下的子树可能有两种情况：
            # 一种是subtree，这时候该rootValue对应的value仍然是一个dict，所以还需要递归下去
            if type(subtreeSet[rootValue]).__name__ == "dict":
                classLabel = classifyDecisionTree(subtreeSet[rootValue], featureNames, inputList)
            # 另一种是叶子节点，这时候对应的value就是一个类标签
            else:
                classLabel = subtreeSet[rootValue]
    return classLabel

# 下面这部分代码用于绘制已经创建好的decision tree的图形
