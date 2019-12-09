
from math import log
import operator
import matplotlib.pyplot as plt


def createDataSet():
    """
    这个函数只是用于生成一个测试数据
    返回值：
    dataSet:一个nested list，每一个sublist包括两部分，前面都是对应于feature的取值，1-yes,0-no,最后一个表示class label
    featureNames：list,表示的是sublist中各个feature的名称，因为dataSet中没有存储feature的名称，只存储了值
    """
    dataSet = [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
    featureNames = ['no surfacing', 'flippers']
    return dataSet, featureNames


def Entropy(data):
    """
    这个函数用于计算当前数据集data的香农熵，而不是直接计算使得information gain最大的feature

    data:是一个nested list，长度就是训练集的大小,其中每一个元素是一个sub-list，sub-list的最后一个元素是classLabel
    这里重要的是sub-list里的类标签(也就是最后一个元素)，sub-list中其他feature的值这里用不上
    shanno:返回值
    """
    # 首先计算entry的数量
    numEntries=len(data)
    # 统计data中类的个数,以及每一类的数量，用dict存储
    classLabels={}
    for featureList in data:
        label=featureList[-1]
        classLabels[label]=classLabels.get(label,0)+1
    # 开始计算信息熵
    shannon=0
    for classCounts in classLabels.values():
        # prob是每一类的比例，这个比例作为概率分布看待
        prob=float(classCounts)/numEntries
        shannon-=prob*log(prob,2)
    return shannon


def filterDataset(dataSet,axis,value):
    """
    这个函数只是将dataSet根据指定的feature和相应的值value进行过滤，只适用于离散feature
    输入：
    dataSet：是一个nested list，每一个sub-list就是一个观测，sub-list的最后一个item就是class label
    axis：用于指定feature的index，int，也就是表示该feature在sub-list中的位置
    value：是axis指定的feature的值
    返回值：
    是一个新的nested list，只包含axis=value的sub-list，并且其中本次使用过的feature被剔除
    """
    newData=[]
    for sublist in dataSet:
        if sublist[axis]==value:
            # 下面这两句将sublist中位于axis处的元素去掉
            reduceList=sublist[:axis]
            # 注意，这里使用的是extend
            reduceList.extend(sublist[axis+1:])
            newData.append(reduceList)
    return newData


def chooseBestFeature(dataSet):
    """
    计算当前dataSet下进行分割的最佳feature，准则是information gain
    输入:
    dataSet:是一个nested list，每一个sub-list是一个观测，并且sub-list的最后一个元素是class label
    这里的feature是用它对应于sub-list中的位置index来表示的
    返回值:
    bestFeature:最佳feature的index，int

    """
    # 计算观测个数
    numEntries=len(dataSet)
    # 计算feature的个数，减去1是因为sub-list的最后一项是class label
    numFeatures=len(dataSet[0])-1
    # 计算dataSet本身的香农熵,其实这个值不用计算，因为对于每个feature来说，都是一样的
    baseShannon=Entropy(dataSet)
    # 其他初始化
    bestInfoGain=0.0
    bestFeature=-1
    for i in range(numFeatures):
        # 找出axis=i的feature所取的value集合
        valuesFeature=[sublist[i] for sublist in dataSet]
        uniqueValue=set(valuesFeature)
        featureEntropy=0.0
        # 根据feature i的各种取值对dataSet进行过滤分割
        for value in uniqueValue:
            # 找出feature i取值为 value的subset
            subset=filterDataset(dataSet,i,value)
            # 计算这个subset的probability，也就是feature i=value的观测个数占整个dataSet的比例
            prob=float(len(subset))/numEntries
            # 计算这个subset的Entropy，这个和计算整个dataSet的Entropy是一样的
            featureEntropy+=prob*Entropy(subset)
        infoGain=baseShannon-featureEntropy
        if infoGain>=bestInfoGain:
            bestInfoGain=infoGain
            bestFeature=i
    return bestFeature


def majorityVote(dataSet):
    """
    这个函数用于处理所有feature都用完时，如何确认leaf node的class label，这里采用多数表决的准则
    输入：
    dataSet:叶子节点的观测集合,nested list，这个函数只会在叶子节点上使用(不进行剪枝时)
    输出：

    """
    classCounts={}
    for sublist in dataSet:
        # 实际上，所有feature都用于进行split时，dataSet中的sublist就只剩下一个class label了，不需要[-1]
        classCounts[sublist[-1]]=classCounts.get(sublist[-1],0)+1
    sortedClassCounts=sorted(classCounts.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCounts[0][0]


def createDecisionTree(dataSet,featureNames):
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
    classLabels=[entries[-1] for entries in dataSet]
    # 判断递归是否停止有两个情况：
    # 情况一：当前dataSet中所有的entries都属于同一个类，那就直接返回这个类标签
    if classLabels.count(classLabels[0])==len(classLabels):
        return classLabels[0]
    # 情况二：当前dataSet中所有的feature都已经被使用过了，那就采取多数表决原则决定类标签（调用上述的majorityVote）
    # 所有feature都被使用过时，dataSet中的sublist就只有一个值——classLabel了，所以长度==1
    if len(dataSet[0])==1:
        return majorityVote(dataSet)
    #上述两个都不满足，说明还可以再进行分割

    # 首先找出当前dataSet下的最佳分割feature(index表示)
    bestFeatureIndex=chooseBestFeature(dataSet)
    # 最佳feature的名称
    bestFeatureName=featureNames[bestFeatureIndex]

    # 使用这个最佳feature构建一棵树，
    # 先用这个feature作为子树的根节点
    deciTree={bestFeatureName:{}}

    # 删除已经使用过的feature名称
    # 这里要特别注意，不能使用下面这句，因为featureNames是作为list传进来的，这里是pass-by-reference
    # 直接删除会导致外面的list内容也没有了
    # del featureNames[bestFeatureIndex]
    # 应当先获取copy
    featureNames=featureNames[:]
    del featureNames[bestFeatureIndex]

    # 统计最佳feature的取值个数
    featureValues=[entries[bestFeatureIndex] for entries in dataSet]
    uniqueValue=set(featureValues)
    # 对于最佳feature的每个取值，分别进行切割，同时构建子树
    for value in uniqueValue:
        # 根据最佳feature的取值,得到feature=value的训练集合subset
        subset=filterDataset(dataSet,bestFeatureIndex,value)
        # 对于这个subset，递归创建子树，注意，这里使用的featureNames已经去掉了使用过的最佳feature的名称
        deciTree[bestFeatureName][value]=createDecisionTree(subset,featureNames)
    return deciTree


def classifyDecisionTree(deciTree,featureNames,inputList):
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
    rootName=list(deciTree.keys())[0]
    rootIndex=featureNames.index(rootName)
    # 根节点之下的子树就是root这个key对应的value——它也是一个nested dict,
    # 不过它是子树的集合(不是单独的一棵子树)，而是root的每一个value都对应于一棵子树，所以是集合
    subtreeSet=deciTree[rootName]
    # 对于根据root创建的subtreeSet，它的key不再是feature了，而是上一个feature root的不同取值，
    # 每个不同的取值对应于不同的subtree——这个才是单独的一棵子树，它的key只有一个，就是这棵子树的root
    for rootValue in subtreeSet.keys():
        if inputList[rootIndex]==rootValue:
            # 根节点root不同的value下的子树可能有两种情况：
            # 一种是subtree，这时候该rootValue对应的value仍然是一个dict，所以还需要递归下去
            if type(subtreeSet[rootValue]).__name__=="dict":
                classLabel=classifyDecisionTree(subtreeSet[rootValue],featureNames,inputList)
            # 另一种是叶子节点，这时候对应的value就是一个类标签
            else:
                classLabel=subtreeSet[rootValue]
    return classLabel


# 下面这部分代码用于绘制已经创建好的decision tree的图形
