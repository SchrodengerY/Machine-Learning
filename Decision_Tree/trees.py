from math import log

def calcShannonEnt(dataSet):

# 计算香农熵(Shannon Entropy)的函数
	numEntries = len(dataSet)
	# 计算数据维度
	labelCounts = {}
	# 定义一个字典
	for featVec in dataSet:
		currentLabel = featVec[-1]
		# 提取特征值
		if currentLabel not in labelCounts.keys():
		# 判断该特征值未包含在字典键值中
			labelCounts[currentLabel] = 0
			# 该键值对应内容赋值为0
		labelCounts[currentLabel] += 1
		# 该特征值包含在字典键值中，则对应内容+1
	shannonEnt = 0.0
	# 初始化香农熵为0
	for key in labelCounts:
		prob = float(labelCounts[key])/numEntries
		# 计算概率
		shannonEnt -= prob * log(prob,2)
		# 计算shannon熵
	return shannonEnt

def createDataSet():
	dataSet = [[1, 1, 'yes'],
			   [1, 1, 'yes'],
			   [1, 0, 'no'],
			   [0, 1, 'no'],
			   [0, 1, 'no']]
	# dataset意义：不浮出水面是否可以生存（1-是；0-否）；是否有脚蹼（1-是；0-否）；属于鱼类
	labels = ['no surfacing', 'flippers']
	# 两类标签：不浮出水面是否可以生存，有脚蹼
	return dataSet, labels


# 相关测试-1
myDat, labels = createDataSet()
print("\n数据集：",myDat)
myShannonEnt = calcShannonEnt(myDat)
# 计算香农熵，熵越高，混合的数据越多。可以通过在数据集中添加更多的分类，观察熵的变化情况
print("\n香农熵：",myShannonEnt)

# 相关测试-2
# 增加了名为maybe的分类
myDat[0][-1] = 'maybe'
print("\n新数据集：",myDat)
myShannonEnt = calcShannonEnt(myDat)
print("\n新香农熵：",myShannonEnt)
# 运行程序后发现，香农熵增大，表明数据集的无序程度增大，即添加了新的分类后会增大数据集的无序程度
# 原因很简单，因为可能性变多了，因此更加难以预测。


# 按照给定特征划分数据集
# 可以理解为寻找一个超平面划分数据，这一点和SVM的思想十分类似。
# SVM学习请移步-July大神的CSDN：http://blog.csdn.net/v_july_v/article/details/7624837

def splitDataSet(dataSet, axis, value):

# 输入的三个参数为：待划分的数据集、划分数据集的特征、需要返回的特征的值
	retDataSet = []
	# 新建了一个返回特征值列表
	for featVec in dataSet:
	# 对于待划分数据集中的每个数据
		if featVec[axis] == value:
		# 如果数据集对应特征的值 = 返回特征的值
			reducedFeatVec = featVec[:axis]
			# 构造一个名为reducedFeatVec的列表（这行语句可以使用下面的语句替代）
			# reducedFeatVec = []
			reducedFeatVec.extend(featVec[axis+1:])
			# 将该特征后面的部分添加进列表
			retDataSet.append(reducedFeatVec)
			# 扩展列表retDataSet

			# 注：extend和append的用法之间有区别
			# extend将多个列表合并为一个单级列表，i.e. a = [1,2,3] b = [4,5,6] a.extend(b) = [1,2,3,4,5,6]
			# append将多个列表合并为一个多级列表. i.e. a = [1,2,3] b = [4,5,6] a.append(b) = [1,2,3,[4,5,6]]

			# 详细python3基础教程请移步-廖大的网站：http://blog.csdn.net/v_july_v/article/details/7624837
	return retDataSet

# 相关测试
myDat, label = createDataSet()
print("\nvalue=1的划分结果：", splitDataSet(myDat, 0, 1))
print("\nvalue=0的划分结果：", splitDataSet(myDat, 0, 0))
# 通过这个测试可以很方便的理解上面函数的意义


# 选择最好的数据集划分方式
# 所采用的方式为：循环遍历整个数据集，循环计算shannon熵和splitDataSet()函数，找到最大的熵所对应的特征划分方式

def chooseBestFeatureToSplit(dataSet):

# 此函数采用了ID3算法，返回最大信息增益所对应的特征划分索引值
# 这种算法有个缺点，信息增益的值是相对于训练数据集而言的，当shannon熵大的时候，信息增益值往往会偏大，这样对shannon熵小的特征不公平。

# 首先，分清楚划分特征和返回特征
# 划分特征：用于对数据集进行划分的某一列元素
# 返回特征：用来和划分特征进行比较的特征，如果返回特征=划分特征，则进行添加操作（见splitDataSet()函数）

	numFeatures = len(dataSet[0]) - 1
	# dataSet（假设为NxM的矩阵）数据集，len(dataSet[0])=M(列数)，numFeatures为[行数-1](原因：因为最后一列是分类，不属于特征)
	# dataSet必须是由列表组成的列表,i.e.[[],[],[]]
	baseEntropy = calcShannonEnt(dataSet)
	# 计算初始shannon熵
	bestInfoGain= 0.0; bestFeature = -1
	# 初始化最大信息增益和最佳特征
	for i in range(numFeatures):
	# 对不同列进行操作（遍历过程）
		featList = [example[i] for example in dataSet]
		# 提取数据集中每个元素的第i个特征（因为第i个特征被用作划分数据集特征，所以需要提取）
		uniqueVals = set(featList)
		# 将featList列表中的元素拆分，放入集合(set)。set的详细用法：http://www.iplaypy.com/jichu/set.html
		# 由于set中不包括重复元素，因此可以最快地得到列表中唯一元素值！
		newEntropy = 0.0
		# 初始化shannon熵
		for value in uniqueVals:
			subDataSet = splitDataSet(dataSet, i, value)
			# 从数据集中提取出子集，其返回特征value来自于set()中。
			prob = len(subDataSet)/float(len(dataSet))
			# 概率计算：极大似然估计概率。基本原理来自于朴素贝叶斯：http://www.hankcs.com/ml/naive-bayesian-method.html
			newEntropy += prob * calcShannonEnt(subDataSet)
			# 计算条件熵
		infoGain = baseEntropy - newEntropy
		# 计算信息增益：shannon熵 - 条件熵

		# infoGainRate = (baseEntropy - newEntropy)/baseEntropy
		# 采用最大信息增益比infoGainRate可以避免最大信息增益带来的不公平问题，此为C4.5算法

		if(infoGain > bestInfoGain):
			bestInfoGain = infoGain
			bestFeature = i
		# 找到最大的信息增益，并将最佳特征值置为i
	return bestFeature

# 递归构造决策树
# 基本工作原理：得到原始数据集，然后基于最好的属性值划分数据集，由于特征值可能多于两个，因此可能存在大于两个分支的数据集划分。
# 		    第一次划分之后，数据将被向下传递到树分支的下一个节点，在这个节点上，我们可以再次划分数据。

# 递归结束条件：程序遍历玩所有划分数据集的属性，或者每个分支下的所有实例都具有相同的分类。
#  			 如果所有实例具有相同的分类，则得到一个叶子节点或者终止块。

# 相关优化决策：如果数据集已经处理了所用属性，但是类标签依然不是唯一的，此时我们需要决定如何定义该叶子节点。
#  		     在这种情况下，通常采用【多数表决】的方法决定该叶子节点的分类

import operator

def majorityCnt(classList):

# 多数表决函数，类似于kNN的分类器
# kNN分类器传送门：https://github.com/SchrodengerY/Machine-Learning/blob/kNN/KNN/kNN.py
# 该函数对没有确定类标签的数据进行判决，并返回出现次数最多的类标签
	classCount = {}
	# 构造类标签统计字典，用于统计不同类标签出现的次数
	for vote in classList:
		if vote not in classCount.keys():
			classCount[vote] = 0
			# 设置一个新的类标签
		classCount[vote] += 1
		# 类标签+1。
	sortedClassCount = sorted(classCount.items(),
							  key = operator.itemgetter(1), reverse = True)
	# 降序排序(reverse = True)类标签字典
	# .items()用于遍历字典中的元素
	return sortedClassCount[0][0]
	# 返回字典中出现次数最多的分类名称

def createTree(dataSet, labels):

# 创建树，dataSet-数据集，labels-标签列表。
	classList = [example[-1] for example in dataSet]
	# 将数据集每个元素的最后一列存入列表（最后一列为标签列）
	if classList.count(classList[0]) == len(classList):
		return classList[0]
	# 如果类别完全相同，则停止继续划分，返回该类标签
	if len(dataSet[0]) == 1:
		return majorityCnt(classList)
	# 如果遍历完所有特征时则返回出现次数最多的分类名称
	bestFeat = chooseBestFeatureToSplit(dataSet)
	bestFeatLabel = labels[bestFeat]
	# 如果无法返回次数最多的分类名称，则挑选出现次数最多的类别作为返回值

	myTree = {bestFeatLabel:{}}
	# 创建一个以类别作为键值的字典
	del(labels[bestFeat])
	# 删除labels中的bestFeat，这是因为这个特征已经作为叶子节点用掉了。
	featValues = [example[bestFeat] for example in dataSet]
	# 得到列表包含的所有属性值
	uniqueVals = set(featValues)
	# 将其定义为集合形式（不包含重复元素）
	for value in uniqueVals:
		subLabels = labels[:]
		# 复制类标签，并将其存储在新列表变量subLabels中
		# 这样做的原因是：python语言中函数参数是列表类型时，参数是按照引用方式传递的。
		# 为了保证每次调用createTree()时不改变原始列表的内容，使用新变量subLabels代替原始列表。
		myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value),
												  subLabels)
	return myTree

# 测试
myDat, labels = createDataSet()
myTree = createTree(myDat, labels)
print("\n决策树：",myTree)