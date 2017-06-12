from numpy import *
import operator

def createDataSet():
	group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
	labels = ['A','A','B','B']
	return group, labels

# inX：用于分类的输入向量；dataSet：输入的训练样本集；labels：标签向量；k：用于选择最近邻居的数目
def classify0(inX, dataSet, labels, k):
	# 读取矩阵第一维的长度，参考网站：http://www.cnblogs.com/zuizui1204/p/6511050.html
	dataSetSize = dataSet.shape[0]
	# tile:重复维度，进行扩展。参考网站：http://blog.csdn.net/wy250229163/article/details/52453201
	# diffMat：表示输入向量和训练样本集之间的距离
	diffMat = tile(inX, (dataSetSize, 1)) - dataSet
	# 分别计算平方:a^2,b^2（d=sqrt(a^2+b^2)）
	sqDiffMat = diffMat**2
	# 将矩阵的每一行元素相加：a^2+b^2。若axis=0则为普通求和
	sqDistances = sqDiffMat.sum(axis = 1)
	# 计算距离开根号
	distances = sqDistances**0.5
	# 按照距离排序，并返回索引
	sortedDistIndices = distances.argsort()
	classCount = {}
	# 选择距离最小的k个点
	for i in range(k):
		# 获取label值
		voteIlabel = labels[sortedDistIndices[i]]
		# 计算label出现次数
		classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
	# 按照label出现的多少排序，降序排列。注：python3中使用items替代iteritems
	sortedClassCount = sorted(classCount.items(),
							  key = operator.itemgetter(1), reverse = True)
	return sortedClassCount[0][0]

# 测试
group, labels = createDataSet()
# 分成两类时k一般取3，这样一定可以进行划分（k取偶数时可能会导致label中出现对等的情况）
print(classify0([0,0], group, labels, 2))
print(classify0([0,0], group, labels, 3))
print(classify0([0,0], group, labels, 4))


# 示例：使用KNN改进约会网站的配对效果

# ------------文本文件转为矩阵-------------
def file2matrix(filename):
	fr = open(filename)
	# 读取文本
	arrayOLines = fr.readlines()
	# 计算文本行数
	numberOfLines = len(arrayOLines)
	# 构造一个零矩阵，维数为：行数*3
	returnMat = zeros((numberOfLines, 3))
	# 构造一个标签向量，用于返回标签值
	classLabelVector = []
	# 构造一个index，用于选择矩阵的行
	index = 0
	for line in arrayOLines:
		# 去除回车符
		line = line.strip()
		# 用\t作为文件分隔位,将字符串进行分割
		listFromLine = line.split('\t')
		# 填充矩阵
		returnMat[index,:] = listFromLine[0:3]
		# 将列表的最后一列添加到标签
		classLabelVector.append(int(listFromLine[-1]))
		# 处理下一行
		index += 1
	return returnMat, classLabelVector

# ---------------导入数据-----------------
datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
print(datingDataMat)
print(datingLabels[0:20])

# ---------------分析数据----------------
# 使用Matplotlib创建散点图

# matplotlib安装：
# 1. sudo apt-get install libfreetype6-dev libxft-dev
# 2. sudo pip3 install matplotlib
import matplotlib
# 显示散点图
matplotlib.use('Qt5Agg')
# matplotlib.matplotlib_fname()
import matplotlib.pyplot as plt

fig = plt.figure()
# 设置图片显示位置信息
ax = fig.add_subplot(111)
# 设置图片散点信息
ax.scatter(datingDataMat[:,1], datingDataMat[:,2])
plt.show()

# 调用scatter函数个性化标记散点图上的点
fig1 = plt.figure()
ax = fig1.add_subplot(111)
ax.scatter(datingDataMat[:,1], datingDataMat[:,2],
		   15.0*array(datingLabels), 15.0*array(datingLabels))
plt.show()


# -------------准备数据------------
# 数据归一化处理
def autoNorm(dataSet):
	# 列最小值(使用min(0)可以从列中选择最小值;使用min(1)可以从行中选择最小值)
	minVals = dataSet.min(0)
	# 列最大值
	maxVals = dataSet.max(0)
	# 区间
	ranges = maxVals - minVals
	# 构造零向量
	normDataSet = zeros(shape(dataSet))
	# 确定行数
	m = dataSet.shape[0]
	# 行列式相减(注：采用行列式的形式替代循环，可以提升速度)
	normDataSet = dataSet - tile(minVals, (m,1))
	# 行列式相除(优点同上)
	normDataSet = normDataSet/tile(ranges, (m,1))
	return normDataSet, ranges, minVals


# 测试算法
def datingClassTest():
	# 此处代码之前有讲解
	hoRatio = 0.10
	datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
	normMat, ranges, minVals = autoNorm(datingDataMat)
	m = normMat.shape[0]
	# 测试数据为10%
	numTestVecs = int(m*hoRatio)
	# 初始化错误率
	errorCount = 0.0
	# 针对每一个测试数据，判断其分类情况
	for i in range(numTestVecs):
		# 利用分类器训练，训练数据为90%
		classifierResult = classify0(normMat[i,:], normMat[numTestVecs:m,:],
									 datingLabels[numTestVecs:m], 3)
		print("the classifier came back with: %d, the real answer is: %d"
			  % (classifierResult, datingLabels[i]))
		if (classifierResult != datingLabels[i]):
			errorCount += 1.0
	print("the total error rate is: %f" % (errorCount/float(numTestVecs)))

# 输出结果
datingClassTest()